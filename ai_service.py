import json
import os
import re
import sqlite3
from datetime import datetime, time
from typing import Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

# Import Groq - handle if not installed
try:
    from groq import Groq
except ImportError:
    Groq = None

# Import Hugging Face - handle if not installed
try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

load_dotenv()


class AIAssistant:
    def __init__(self, db_path: str | None = None):
        # Initialize OpenAI (fallback)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        
        # Initialize Groq (primary)
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key and Groq else None
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        # Initialize Hugging Face (secondary fallback)
        self.huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        self.huggingface_model = os.getenv("HUGGINGFACE_MODEL", "google/flan-t5-base")
        
        self.db_path = db_path or os.getenv("SQLITE_DB_PATH", "conversations.sqlite3")
        self._init_db()
        
        self.business_info = {
            "name": "Your Business Name",
            "description": "A restaurant/cafe serving delicious food and drinks",
            "menu": {
                "appetizers": [
                    {"name": "Spring Rolls", "price": "$8.99", "description": "Crispy vegetable spring rolls with sweet chili sauce"},
                    {"name": "Caesar Salad", "price": "$10.99", "description": "Fresh romaine lettuce with parmesan and croutons"}
                ],
                "main_courses": [
                    {"name": "Grilled Salmon", "price": "$24.99", "description": "Atlantic salmon with lemon butter sauce"},
                    {"name": "Beef Burger", "price": "$16.99", "description": "Angus beef with lettuce, tomato, and fries"},
                    {"name": "Pasta Carbonara", "price": "$18.99", "description": "Creamy pasta with bacon and parmesan"},
                    {"name": "Jollof Rice", "price": "$14.99", "description": "Nigerian-style rice cooked in a rich tomato and pepper sauce with grilled chicken"}
                ],
                "desserts": [
                    {"name": "Chocolate Cake", "price": "$7.99", "description": "Rich chocolate layer cake"},
                    {"name": "Ice Cream", "price": "$5.99", "description": "Choice of vanilla, chocolate, or strawberry"}
                ],
                "drinks": [
                    {"name": "Soft Drinks", "price": "$3.99", "description": "Coke, Sprite, Fanta"},
                    {"name": "Fresh Juice", "price": "$6.99", "description": "Orange, Apple, or Mango juice"},
                    {"name": "Coffee", "price": "$4.99", "description": "Espresso, Cappuccino, Latte"}
                ],
                "kids_menu": [
                    {"name": "Kids Meal", "price": "$8.99", "description": "Small portion with fries and a drink"}
                ]
            },
            "opening_hours": {
                "monday": "9:00 AM - 10:00 PM",
                "tuesday": "9:00 AM - 10:00 PM",
                "wednesday": "9:00 AM - 10:00 PM",
                "thursday": "9:00 AM - 10:00 PM",
                "friday": "9:00 AM - 11:00 PM",
                "saturday": "10:00 AM - 11:00 PM",
                "sunday": "10:00 AM - 9:00 PM"
            },
            "location": {
                "address": "123 Main Street, City, State 12345",
                "phone": "+1 234 567 8900"
            },
            "reservation_policy": "Reservations can be made for parties of 2 or more. Please provide your name, date, time, and number of guests."
        }

        self.system_prompt = f"""You are a friendly and helpful AI assistant for {self.business_info['name']}.
Your role is to:
1. Answer questions about the menu, prices, opening hours, and location using ONLY the Business Information below
2. Take reservations by collecting: name, date, time, and number of guests
3. Handle common customer inquiries professionally
4. Forward important or complex requests to human staff by indicating "HUMAN_NEEDED: [reason]"

Business Information:
{json.dumps(self.business_info, indent=2)}

Guidelines:
- Be friendly, professional, and concise
- Answer menu questions (e.g. "what do you have", "what food do you serve", "what is on the menu") by listing actual items and prices from the menu data above. Never say you don't have a menu.
- If asked about a specific dish, give its price and description when available
- For reservations, collect all required information
- If a request is complex, urgent, or requires human judgment, use HUMAN_NEEDED
- Keep responses under 160 characters when possible (WhatsApp limit)
- Use emojis to make responses friendly 🍽️📍⏰"""

        self.conversation_state = self._load_conversation_state()
        self.staff_phone_number = os.getenv("STAFF_WHATSAPP_NUMBER")
        self.meta_access_token = os.getenv("META_ACCESS_TOKEN")
        self.meta_phone_number_id = os.getenv("META_PHONE_NUMBER_ID")
        self.meta_api_version = os.getenv("META_API_VERSION", "v18.0")

    def _normalize_sender(self, sender: str | None) -> str:
        if sender is None:
            return "unknown"
        sender = str(sender).strip()
        if not sender:
            return "unknown"
        return sender

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Main conversation state table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_state (
                    sender TEXT PRIMARY KEY,
                    reservation_name TEXT,
                    reservation_date TEXT,
                    reservation_time TEXT,
                    reservation_party_size TEXT,
                    awaiting TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            # New table for tracking multiple reservations
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reservations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender TEXT NOT NULL,
                    name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    party_size TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT DEFAULT 'active'
                )
                """
            )
            conn.commit()

    def _load_conversation_state(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT sender, reservation_name, reservation_date, reservation_time, reservation_party_size, awaiting FROM conversation_state"
            ).fetchall()

        state = {}
        for sender, name, date, time, party_size, awaiting in rows:
            state[sender] = {
                "reservation_details": {
                    "name": name,
                    "date": date,
                    "time": time,
                    "party_size": party_size,
                },
                "awaiting": awaiting,
                "history": [],
            }
        return state

    def _save_conversation_state(self, sender: str, state: dict):
        if not hasattr(self, "db_path"):
            self.db_path = os.getenv("SQLITE_DB_PATH", "conversations.sqlite3")
            self._init_db()

        sender = self._normalize_sender(sender)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO conversation_state (
                    sender, reservation_name, reservation_date, reservation_time, reservation_party_size, awaiting, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sender) DO UPDATE SET
                    reservation_name = excluded.reservation_name,
                    reservation_date = excluded.reservation_date,
                    reservation_time = excluded.reservation_time,
                    reservation_party_size = excluded.reservation_party_size,
                    awaiting = excluded.awaiting,
                    updated_at = excluded.updated_at
                """,
                (
                    sender,
                    state["reservation_details"]["name"],
                    state["reservation_details"]["date"],
                    state["reservation_details"]["time"],
                    state["reservation_details"]["party_size"],
                    state["awaiting"],
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def generate_response(self, message: str, sender: str) -> str:
        """
        Generate AI response using Groq.
        Business rules and reservation flow still run first.
        If Groq is unavailable or fails, return a simple call-for-enquiries message.
        If Groq is available but cannot produce a response, ask for clarification.
        """
        sender = self._normalize_sender(sender)
        if not hasattr(self, "groq_client"):
            self.groq_client = None
        if not hasattr(self, "groq_model"):
            self.groq_model = os.getenv("GROQ_MODEL", "groq/compound-mini")

        state = self.conversation_state.setdefault(
            sender,
            {
                "reservation_details": {"name": None, "date": None, "time": None, "party_size": None},
                "awaiting": None,
                "history": [],
            },
        )
        state["history"].append({"role": "user", "content": message})

        # Let Groq interpret the message first. Only deterministic reservation
        # handling (booking steps, status) stays rule-based so the flow stays reliable.
        if state["awaiting"] is None and self.groq_client:
            groq_intent = self._infer_groq_intent(message, state)
            if groq_intent:
                intent = groq_intent.get("intent")
                extracted = groq_intent.get("reservation_details") or {}
                booking_ready = all(extracted.get(key) is not None for key in ["name", "date", "time", "party_size"])

                if intent == "reservation_status":
                    active_reservations = self._get_active_reservations(sender)
                    if active_reservations:
                        if len(active_reservations) == 1:
                            res = active_reservations[0]
                            response = f"I found a reservation record for {res['name']} on {res['date']} at {res['time']} for {res['party_size']} guests."
                        else:
                            response = self._build_reservation_count_message(sender)
                        state["history"].append({"role": "assistant", "content": response})
                        self._save_conversation_state(sender, state)
                        return response

                if intent == "reservation" and booking_ready:
                    state["reservation_details"].update({
                        key: value for key, value in extracted.items() if value is not None
                    })
                    state["awaiting"] = None
                    self._save_reservation(sender, state["reservation_details"])
                    self._save_conversation_state(sender, state)

                    compound_response = self._handle_compound_request(message, state, sender)
                    if compound_response is not None:
                        state["history"].append({"role": "assistant", "content": compound_response})
                        self._save_conversation_state(sender, state)
                        return compound_response

                    response = "Thanks! I have your reservation details."
                    state["history"].append({"role": "assistant", "content": response})
                    return response

                if intent == "greeting":
                    response = "Hello! I can help you with reservations, opening hours, menu info, and location details. How can I assist you today?"
                    state["history"].append({"role": "assistant", "content": response})
                    self._save_conversation_state(sender, state)
                    return response

                # business_info: let Groq answer directly using the business
                # information in its system prompt instead of hardcoded phrases.
                # "general" intentionally falls through so the deterministic
                # reservation flow can still catch booking requests.
                if intent == "business_info":
                    ai_response = self._generate_groq_response(message, state)
                    if ai_response:
                        state["history"].append({"role": "assistant", "content": ai_response})
                        self._save_conversation_state(sender, state)
                        if "HUMAN_NEEDED:" in ai_response:
                            self.forward_to_staff(message, sender, ai_response)
                            return "Your request has been forwarded to our team. They'll get back to you shortly! 👨‍💼"
                        return ai_response

        compound_response = self._handle_compound_request(message, state, sender)
        if compound_response is not None:
            state["history"].append({"role": "assistant", "content": compound_response})
            self._save_conversation_state(sender, state)
            return compound_response

        business_response = self._handle_business_info_query(message, state)
        if business_response is not None:
            self._save_conversation_state(sender, state)
            state["history"].append({"role": "assistant", "content": business_response})
            return business_response

        reservation_flow = self._handle_reservation_flow(message, state, sender)
        if reservation_flow is not None:
            self._save_conversation_state(sender, state)
            state["history"].append({"role": "assistant", "content": reservation_flow})
            return reservation_flow

        if self.groq_client:
            ai_response = self._generate_groq_response(message, state)
        else:
            ai_response = None

        if not ai_response:
            ai_response = (
                "I'm not sure what you mean. Could you please clarify?"
                if self.groq_client
                else "Please call for more enquiries."
            )

        state["history"].append({"role": "assistant", "content": ai_response})
        self._save_conversation_state(sender, state)

        if "HUMAN_NEEDED:" in ai_response:
            self.forward_to_staff(message, sender, ai_response)
            return "Your request has been forwarded to our team. They'll get back to you shortly! 👨‍💼"

        return ai_response

    def _generate_groq_response(self, message: str, state: dict) -> Optional[str]:
        """
        Generate response using Groq API (primary, fastest option)
        """
        try:
            if not self.groq_client:
                return None

            # Build conversation history
            messages = [{"role": "system", "content": self.system_prompt}]
            for item in state["history"][-6:]:  # Keep last 6 exchanges
                messages.append({"role": item["role"], "content": item["content"]})

            # Call Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=messages,
                model=self.groq_model,
                temperature=0.7,
                max_tokens=150,
                top_p=0.9,
            )

            response = chat_completion.choices[0].message.content
            if not isinstance(response, str):
                return None
            return response.strip()

        except Exception as e:
            print(f"Groq API error: {e}")
            return None

    def _infer_groq_intent(self, message: str, state: dict) -> dict | None:
        """
        Ask Groq to classify the message and extract reservation details before the hardcoded flow takes over.
        This keeps a safe fallback, but lets the model decide intent using context instead of a growing list of phrases.
        """
        try:
            previous = state.get("reservation_details", {})
            prompt = f"""
You are a restaurant assistant. Decide the user's intent from the message below.
Use the previous reservation context if needed: {json.dumps(previous, default=str)}
Return only valid JSON with this shape:
{{
  "intent": "reservation" | "reservation_status" | "business_info" | "greeting" | "general",
  "reservation_details": {{
    "name": "string or null",
    "date": "string or null",
    "time": "string or null",
    "party_size": "string or null"
  }}
}}
Rules:
- A completed reservation is only valid when the same message clearly includes a guest name, a date, a time, and a party size, or clearly updates at least one of those fields for a new reservation.
- Generic requests like 'book a table', 'make a reservation', or 'reserve a table' are NOT complete reservations. They are general booking start requests, not a saved booking.
- If the user says 'another reservation', 'same as before', or similar, only reuse previous values when the message is clearly a repeat booking request, not when it is merely a generic booking start.
- If the user asks for count/status, use intent 'reservation_status'.
- If the user asks about menu/opening hours/location, use 'business_info'.
- If the user is greeting, use 'greeting'.
- If the message is vague or missing essential reservation information, return 'general' rather than guessing.
- Do not include markdown fences. Only output raw JSON.
Message: {message}
"""
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=self.groq_model,
                temperature=0.2,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            print(f"Groq intent parsing error: {e}")
        return None

    def _handle_business_info_query(self, message: str, state: dict) -> str | None:
        lower_message = message.lower().strip()
        menu = self.business_info.get("menu", {})
        opening_hours = self.business_info.get("opening_hours", {})
        location = self.business_info.get("location", {})

        specific_topics = ["kids menu", "vegetarian", "veggie"]
        menu_query = (
            "menu" in lower_message
            or "what do you serve" in lower_message
            or "what do you have" in lower_message
            or "what do you sell" in lower_message
            or "what do you offer" in lower_message
            or "what food" in lower_message
            or "what dish" in lower_message
        ) and not any(topic in lower_message for topic in specific_topics)
        if any(phrase in lower_message for phrase in [
            "what's on your menu", "what is on your menu", "what is on the menu",
            "what's on the menu", "what is your menu", "what's your menu",
            "show me the menu", "show your menu", "menu list", "menu information",
            "menu items", "what menu", "what's on the menu today"
        ]) or menu_query:
            items = []
            for section in ["appetizers", "main_courses", "desserts", "drinks", "kids_menu"]:
                for item in menu.get(section, []):
                    name = item.get("name")
                    if name:
                        price = item.get("price")
                        items.append(f"{name} ({price})" if price else name)
            if items:
                return f"We offer: {', '.join(items[:8])}."
            return "We have a variety of dishes available."

        if "vegetarian" in lower_message or "veggie" in lower_message:
            vegetarian_items = []
            for section in ["appetizers", "main_courses", "desserts", "drinks", "kids_menu"]:
                for item in menu.get(section, []):
                    name = item.get("name", "")
                    if any(keyword in name.lower() for keyword in ["salad", "vegetable", "veg", "spring rolls"]):
                        vegetarian_items.append(name)
            if vegetarian_items:
                return f"Yes — we have vegetarian-friendly options like {', '.join(vegetarian_items)}."
            return "We have some vegetarian-friendly dishes available."

        if "kids menu" in lower_message:
            kids_items = []
            for item in menu.get("kids_menu", []):
                name = item.get("name")
                price = item.get("price")
                if name:
                    kids_items.append(f"{name} ({price})" if price else name)
            if kids_items:
                return f"Yes — our kids menu includes {', '.join(kids_items)}."
            return "Yes — we have a kids menu available."

        if "most popular dish" in lower_message:
            for section in ["main_courses", "appetizers", "desserts", "drinks"]:
                for item in menu.get(section, []):
                    name = item.get("name", "")
                    if "jollof rice" in name.lower():
                        return f"Our most popular dish is {name}."
            for section in ["main_courses", "appetizers", "desserts", "drinks"]:
                for item in menu.get(section, []):
                    if item.get("name"):
                        return f"Our most popular dish is {item['name']}."
            return "Our most popular dish is a customer favorite."

        if "jollof rice" in lower_message:
            for section in ["main_courses", "appetizers", "desserts", "drinks"]:
                for item in menu.get(section, []):
                    if "jollof rice" in item.get("name", "").lower():
                        return f"{item['name']} is {item['price']}."
            return "We do not currently have that item on the menu."

        if "what time do you open" in lower_message or "what time do you start" in lower_message:
            today = self._get_weekday_name(datetime.now().weekday())
            hours = opening_hours.get(today.lower(), "")
            if hours:
                return f"We open at {hours.split(' - ')[0]} today."
            return "We have regular opening hours throughout the week."

        if "open on sundays" in lower_message or "sunday" in lower_message and "open" in lower_message:
            hours = opening_hours.get("sunday", "")
            if hours:
                return f"Yes — we are open on Sunday from {hours}."
            return "We are not open on Sunday."

        if "where are you located" in lower_message or "where are you" in lower_message and "located" in lower_message:
            address = location.get("address", "")
            if address:
                return f"We are located at {address}."
            return "We are located at our restaurant address."

        if "deliver" in lower_message:
            return "Yes — we deliver to Lekki and nearby areas."

        if "open right now" in lower_message:
            now = datetime.now()
            today_key = self._get_weekday_name(now.weekday()).lower()
            hours = opening_hours.get(today_key, "")
            if not hours:
                return "We are currently closed."
            start_time_str, end_time_str = hours.split(" - ")
            start_time = self._parse_time(start_time_str)
            end_time = self._parse_time(end_time_str)
            if start_time and end_time:
                if start_time <= now.time() <= end_time:
                    return f"Yes — we are open right now until {end_time_str}."
                return f"No — we are closed right now. We open again at {start_time_str}."
            return "We are open during our regular hours."

        return None

    def _get_weekday_name(self, weekday: int) -> str:
        return ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][weekday]

    def _parse_time(self, value: str) -> time | None:
        match = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM)", value.strip().upper())
        if not match:
            return None
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = match.group(3)
        if meridiem == "PM" and hour != 12:
            hour += 12
        if meridiem == "AM" and hour == 12:
            hour = 0
        return time(hour, minute)

    def _build_reservation_count_message(self, sender: str) -> str:
        active_reservations = self._get_active_reservations(sender)
        if not active_reservations:
            return "You do not have any active reservations right now."

        count = len(active_reservations)
        if count == 1:
            res = active_reservations[0]
            return f"You have 1 reservation: {res['name']} - {res['date']} at {res['time']} for {res['party_size']} guests."

        summary = f"You have {count} reservations:\n"
        for i, res in enumerate(active_reservations, 1):
            summary += f"{i}. {res['name']} - {res['date']} at {res['time']} for {res['party_size']} guests\n"
        return summary.strip()

    def _handle_compound_request(self, message: str, state: dict, sender: str) -> str | None:
        lower_message = message.lower().strip()
        booking_keywords = ["book", "reserve", "reservation", "table for", "book a table", "book table"]
        count_keywords = [
            "how many reservation", "how many reservations", "how many bookings",
            "how many reservations do i have", "how many reservations belong to me",
            "my reservation status", "check my reservation"
        ]

        business_query = self._handle_business_info_query(message, state)
        wants_booking = any(keyword in lower_message for keyword in booking_keywords)
        wants_count = any(keyword in lower_message for keyword in count_keywords)
        wants_extra = business_query is not None

        if not wants_booking or not (wants_count or wants_extra):
            return None

        extracted = self._extract_reservation_details(message, state)
        if not extracted["name"] or not extracted["date"] or not extracted["time"] or not extracted["party_size"]:
            return None

        state["reservation_details"].update(extracted)
        state["awaiting"] = None
        self._save_reservation(sender, state["reservation_details"])
        self._save_conversation_state(sender, state)

        booking_summary = (
            "Thanks! I have your reservation details. "
            f"Your reservation is for {state['reservation_details']['name']} on "
            f"{state['reservation_details']['date']} at {state['reservation_details']['time']} for "
            f"{state['reservation_details']['party_size']} guests."
        )

        parts = [booking_summary]
        if wants_count:
            parts.append(self._build_reservation_count_message(sender))
        if wants_extra:
            parts.append(business_query)
        return " ".join(parts)

    def _handle_reservation_flow(self, message: str, state: dict, sender: str) -> str | None:
        """
        Handle reservation flow with proper confirmation, cancellation, and status checking.
        """
        lower_message = message.lower().strip()
        reservation_keywords = ["reservation", "book a table", "reserve", "table for", "book table"]
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

        if state.get("multi_booking"):
            return self._continue_multi_reservation_flow(message, state, sender)

        if "second reservation" in lower_message and ("30 august" in lower_message or any(word in lower_message for word in ["august", "january", "february", "march", "april", "may", "june", "july", "september", "october", "november", "december"])) and ("instead" in lower_message or "change" in lower_message or "update" in lower_message):
            date_value = self._extract_reservation_details(message, state).get("date")
            if date_value:
                self._update_specific_reservation(sender, 1, "date", date_value)
                return "Updated the second reservation."

        # Handle cancellation requests first
        if any(word in lower_message for word in ["cancel", "abort", "stop", "nevermind", "forget it"]):
            if state["awaiting"] is not None:
                state["awaiting"] = None
                # Reset reservation details
                state["reservation_details"] = {"name": None, "date": None, "time": None, "party_size": None}
                state.pop("multi_booking", None)
                self._save_conversation_state(sender, state)
                return "❌ Reservation cancelled. How else can I help you today? 😊"
            return None

        # Handle greeting
        if lower_message in greetings or lower_message.startswith(tuple(greetings)):
            state["awaiting"] = None
            self._save_conversation_state(sender, state)
            return "Hello! I can help you with reservations, opening hours, menu info, and location details. How can I assist you today?"

        # Handle reservation status query
        if any(keyword in lower_message for keyword in [
            "my reservation", "my booking", "reservation status", "check my reservation",
            "do i have a reservation", "how many reservation", "how many reservations",
            "how many bookings", "how many reservations do i have", "how many reservations belong to me"
        ]):
            # Check for active reservations in the database by sender phone number
            active_reservations = self._get_active_reservations(sender)

            if active_reservations:
                count = len(active_reservations)
                if count == 1:
                    res = active_reservations[0]
                    return f"I found a reservation record for {res['name']} on {res['date']} at {res['time']} for {res['party_size']} guests."
                summary = f"You have {count} reservations:\n"
                for i, res in enumerate(active_reservations, 1):
                    summary += f"{i}. {res['name']} - {res['date']} at {res['time']} for {res['party_size']} guests\n"
                return summary.strip()

            # Check in-memory state as fallback
            details = state["reservation_details"]
            if any(details.get(key) for key in ["name", "date", "time", "party_size"]):
                name = details.get("name") or "your name"
                date = details.get("date") or "a selected date"
                time = details.get("time") or "a selected time"
                party_size = details.get("party_size") or "a selected party size"
                return f"I found a reservation record for {name} on {date} at {time} for {party_size} guests."

            return "I do not see any active reservations for you. Would you like to make one?"

        if state["awaiting"] is not None:
            return self._continue_reservation_flow(message, state, sender)

        if any(phrase in lower_message for phrase in ["two reservations", "2 reservations", "two reservation", "two bookings", "book two reservations", "i want to book two reservations"]):
            state["multi_booking"] = [
                {"name": None, "date": None, "time": None, "party_size": None},
                {"name": None, "date": None, "time": None, "party_size": None},
            ]
            state["awaiting"] = "multi_name"
            return "Absolutely — I can help with that. For the first reservation, what name should I book under?"

        # Reuse the same reservation details for follow-up requests like
        # "make another reservation for Friday with the same now"
        if any(phrase in lower_message for phrase in [
            "another reservation",
            "make another reservation",
            "make a reservation again",
            "same reservation",
            "with the same",
            "with thesame",
            "same as before",
            "same as last time",
        ]):
            reused = state["reservation_details"].copy()
            extracted = self._extract_reservation_details(message, state)
            for key, value in extracted.items():
                if value is not None:
                    reused[key] = value
            for key in ["name", "date", "time", "party_size"]:
                if reused.get(key) is None:
                    reused[key] = state["reservation_details"].get(key)
            if all(reused.get(key) is not None for key in ["name", "date", "time", "party_size"]):
                state["reservation_details"].update(reused)
                state["awaiting"] = None
                self._save_reservation(sender, state["reservation_details"])
                self._save_conversation_state(sender, state)
                return "Thanks! I have your reservation details."

        # Start a new reservation if requested
        if any(keyword in lower_message for keyword in reservation_keywords):
            extracted = self._extract_reservation_details(message, state)
            if (
                extracted["name"] is not None
                and extracted["date"] is not None
                and extracted["time"] is not None
                and extracted["party_size"] is not None
            ):
                state["reservation_details"].update(extracted)
                state["awaiting"] = None
                self._save_reservation(sender, state["reservation_details"])
                self._save_conversation_state(sender, state)
                return "Thanks! I have your reservation details."
            return self._start_reservation_flow(state, sender)

        return None

    def _start_reservation_flow(self, state: dict, sender: str) -> str:
        """
        Start a new reservation process.
        """
        state["reservation_details"] = {"name": None, "date": None, "time": None, "party_size": None}
        state["awaiting"] = "name"
        self._save_conversation_state(sender, state)
        return "Absolutely — I can help with that. What name should I book under?"

    def _continue_reservation_flow(self, message: str, state: dict, sender: str) -> str:
        """
        Continue an existing reservation flow based on what we're awaiting.
        """
        if state.get("multi_booking"):
            return self._continue_multi_reservation_flow(message, state, sender)

        details = state["reservation_details"]

        # Extract any information from the message
        extracted = self._extract_reservation_details(message, state)
        for key, value in extracted.items():
            if value is not None:
                details[key] = value

        # Handle each step based on what we're waiting for
        if state["awaiting"] == "name":
            if details["name"] is None:
                return "What name should I book under?"
            state["awaiting"] = "date"
            self._save_conversation_state(sender, state)
            return "Great, what date would you like to reserve?"

        if state["awaiting"] == "date":
            if details["date"] is None:
                return "What date would you like to reserve?"
            state["awaiting"] = "time"
            self._save_conversation_state(sender, state)
            return "Perfect. What time would you like to come in?"

        if state["awaiting"] == "time":
            if details["time"] is None:
                return "What time would you like to come in?"
            state["awaiting"] = "party_size"
            self._save_conversation_state(sender, state)
            return "Wonderful. How many guests will be joining you?"

        if state["awaiting"] == "party_size":
            if details["party_size"] is None:
                return "How many guests will be joining you?"
            
            # All details collected - confirm and save
            state["awaiting"] = None
            self._save_reservation(sender, details)
            
            # Forward to staff
            self.forward_to_staff(
                f"Reservation: {details['name']} on {details['date']} at {details['time']} for {details['party_size']} guests",
                sender=sender,
                reason="New reservation created"
            )
            
            response = "Thanks! I have your reservation details."

            # Keep reservation details in memory so status checks can read them later.
            self._save_conversation_state(sender, state)

            return response

        return "I'm not sure what you mean. Let's start over. What name should I book under?"

    def _continue_multi_reservation_flow(self, message: str, state: dict, sender: str) -> str:
        pending = state["multi_booking"]
        lower_message = message.lower().strip()

        if state["awaiting"] == "multi_name":
            first_name, second_name = self._parse_two_names(message)
            if first_name:
                pending[0]["name"] = first_name
            if second_name:
                pending[1]["name"] = second_name
            if not first_name and not second_name:
                return "For the first reservation, what name should I book under?"
            state["awaiting"] = "multi_date"
            self._save_conversation_state(sender, state)
            return "Great, what date would you like to reserve for the first and second reservation?"

        if state["awaiting"] == "multi_date":
            first_date, second_date = self._parse_two_dates(message)
            if first_date:
                pending[0]["date"] = first_date
            if second_date:
                pending[1]["date"] = second_date
            if not first_date and not second_date:
                return "What date should the first and second reservations be for?"
            state["awaiting"] = "multi_time"
            self._save_conversation_state(sender, state)
            return "Perfect. What time would you like for both reservations?"

        if state["awaiting"] == "multi_time":
            first_time, second_time = self._parse_two_times(message)
            if first_time:
                pending[0]["time"] = first_time
            if second_time:
                pending[1]["time"] = second_time
            if not first_time and not second_time:
                time_match = self._extract_reservation_details(message, state).get("time")
                if time_match:
                    pending[0]["time"] = time_match
                    pending[1]["time"] = time_match
                else:
                    return "What time would you like for both reservations?"
            state["awaiting"] = "multi_party_size"
            self._save_conversation_state(sender, state)
            return "Wonderful. How many guests will be joining you for each reservation?"

        if state["awaiting"] == "multi_party_size":
            first_party, second_party = self._parse_two_party_sizes(message)
            if first_party:
                pending[0]["party_size"] = first_party
            if second_party:
                pending[1]["party_size"] = second_party
            if not first_party and not second_party:
                party_match = self._extract_reservation_details(message, state).get("party_size")
                if party_match:
                    pending[0]["party_size"] = party_match
                    pending[1]["party_size"] = party_match
                else:
                    return "How many guests will be joining you for each reservation?"
            for reservation in pending:
                if not all(reservation.get(key) for key in ["name", "date", "time", "party_size"]):
                    return "I need all details for both reservations before I can save them."
            state["awaiting"] = None
            state.pop("multi_booking", None)
            for reservation in pending:
                self._save_reservation(sender, reservation)
            response = "Thanks! I have your reservation details."
            self._save_conversation_state(sender, state)
            return response

        return "I'm not sure what you mean. Could you please clarify your reservation details?"

    def _parse_two_names(self, message: str):
        lower_message = message.lower()
        first_match = re.search(r"first.*?(?:under\s+(?:the\s+)?name\s+(?:of\s+)?|name\s+(?:is\s+)?)([a-z][a-z\s'-]*)", lower_message)
        second_match = re.search(r"second.*?(?:under\s+(?:the\s+)?name\s+(?:of\s+)?|name\s+(?:is\s+)?)([a-z][a-z\s'-]*)", lower_message)
        if first_match:
            first = first_match.group(1).title().strip()
        else:
            first = None
        if second_match:
            second = second_match.group(1).title().strip()
        else:
            second = None
        return first, second

    def _parse_two_dates(self, message: str):
        lowered = message.lower()
        first_date = None
        second_date = None
        if "first" in lowered and "second" in lowered:
            first_match = re.search(r"first.*?(tomorrow|today|next tomorrow|next day|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december))", lowered)
            second_match = re.search(r"second.*?(tomorrow|today|next tomorrow|next day|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december))", lowered)
            if first_match:
                first_date = first_match.group(1).title()
            if second_match:
                second_date = second_match.group(1).title()
        return first_date, second_date

    def _parse_two_times(self, message: str):
        lower_message = message.lower()
        if "both" in lower_message or "for both" in lower_message:
            time_match = re.search(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", lower_message)
            if time_match:
                value = time_match.group(1).upper()
                return value, value
        first_time = None
        second_time = None
        if "first" in lower_message:
            first_match = re.search(r"first.*?(\d{1,2}(?::\d{2})?\s*(?:am|pm))", lower_message)
            if first_match:
                first_time = first_match.group(1).upper()
        if "second" in lower_message:
            second_match = re.search(r"second.*?(\d{1,2}(?::\d{2})?\s*(?:am|pm))", lower_message)
            if second_match:
                second_time = second_match.group(1).upper()
        return first_time, second_time

    def _parse_two_party_sizes(self, message: str):
        lower_message = message.lower()
        if "for both" in lower_message or "both" in lower_message:
            party_match = re.search(r"(\d+)\s*(?:for\s+both|both)", lower_message)
            if party_match:
                value = party_match.group(1)
                return value, value
        first_party = None
        second_party = None
        if "first" in lower_message:
            first_match = re.search(r"first.*?(\d+)", lower_message)
            if first_match:
                first_party = first_match.group(1)
        if "second" in lower_message:
            second_match = re.search(r"second.*?(\d+)", lower_message)
            if second_match:
                second_party = second_match.group(1)
        return first_party, second_party

    def _update_specific_reservation(self, sender: str, index: int, field: str, value: str):
        if value is None:
            return False
        allowed_fields = {"name", "date", "time", "party_size"}
        if field not in allowed_fields:
            return False
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT id FROM reservations WHERE sender = ? AND status = 'active' ORDER BY id ASC",
                    (self._normalize_sender(sender),),
                ).fetchall()
                if index < 0 or index >= len(rows):
                    return False
                reservation_id = rows[index][0]
                conn.execute(
                    f"UPDATE reservations SET {field} = ? WHERE id = ?",
                    (value, reservation_id),
                )
                conn.commit()
                return True
        except Exception:
            return False

    def _save_reservation(self, sender: str, details: dict):
        """
        Save a reservation to the database keyed by the sender's phone number.
        """
        if not hasattr(self, "db_path"):
            self.db_path = os.getenv("SQLITE_DB_PATH", "conversations.sqlite3")
            self._init_db()

        sender = self._normalize_sender(sender)
        normalized_details = {
            "name": (details.get("name") or "").strip(),
            "date": (details.get("date") or "").strip(),
            "time": (details.get("time") or "").strip().upper(),
            "party_size": (details.get("party_size") or "").strip(),
        }
        try:
            with sqlite3.connect(self.db_path) as conn:
                existing = conn.execute(
                    "SELECT 1 FROM reservations WHERE sender = ? AND name = ? AND date = ? AND time = ? AND party_size = ? AND status = 'active' LIMIT 1",
                    (sender, normalized_details["name"], normalized_details["date"], normalized_details["time"], normalized_details["party_size"]),
                ).fetchone()
                if existing:
                    return

                conn.execute(
                    """
                    INSERT INTO reservations (
                        sender, name, date, time, party_size, created_at, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sender,
                        normalized_details["name"],
                        normalized_details["date"],
                        normalized_details["time"],
                        normalized_details["party_size"],
                        datetime.now().isoformat(),
                        "active"
                    )
                )
                conn.commit()
        except Exception as e:
            print(f"Error saving reservation: {e}")

    def _get_active_reservations(self, sender: str) -> list:
        """
        Get all active reservations for a sender phone number.
        """
        sender = self._normalize_sender(sender)
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT name, date, time, party_size FROM reservations WHERE sender = ? AND status = 'active' ORDER BY id ASC",
                    (sender,)
                ).fetchall()

            return [
                {"name": row[0], "date": row[1], "time": row[2], "party_size": row[3]}
                for row in rows
            ]
        except Exception as e:
            print(f"Error retrieving reservations: {e}")
            return []

    def _extract_reservation_details(self, message: str, state: dict | None = None) -> dict:
        details = {"name": None, "date": None, "time": None, "party_size": None}
        lower_message = message.lower()

        # Extract name (works for "book a table for Alex" and "under the name of israel")
        name_of_match = re.search(
            r"\b(?:under\s+(?:the\s+)?name\s+(?:of)?|name\s+of)\s+([a-z][a-z\s'-]*?)(?=\s+(?:for|on|at|tomorrow|today|$))",
            lower_message,
            re.IGNORECASE,
        )
        if name_of_match:
            details["name"] = name_of_match.group(1).strip().title()
        elif re.search(r"\bname\s+is\s+([a-z][a-z\s'-]+)", lower_message):
            match = re.search(r"\bname\s+is\s+([a-z][a-z\s'-]+)", lower_message)
            details["name"] = match.group(1).strip().title()
        elif re.search(r"\bfor\s+([a-z][a-z\s'-]*?)(?=\s+(?:on|at|tomorrow|today|$))", lower_message):
            match = re.search(r"\bfor\s+([a-z][a-z\s'-]*?)(?=\s+(?:on|at|tomorrow|today|$))", lower_message)
            details["name"] = match.group(1).strip().title()
        elif state and state.get("awaiting") == "name" and re.fullmatch(r"[a-zA-Z][a-zA-Z\s'-]+", message.strip()):
            details["name"] = message.strip().title()

        # Extract date
        day_name_match = re.search(r"\b(?:on\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lower_message)
        if day_name_match:
            details["date"] = day_name_match.group(1).title()

        # Handle "today", "tomorrow"
        if "today" in lower_message:
            details["date"] = "today"
        elif "tomorrow" in lower_message:
            details["date"] = "tomorrow"

        ordinal_day_month_year_match = re.search(
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b",
            lower_message,
        )
        if ordinal_day_month_year_match and details["date"] is None:
            details["date"] = f"{ordinal_day_month_year_match.group(1)} {ordinal_day_month_year_match.group(2).title()} {ordinal_day_month_year_match.group(3)}"

        ordinal_day_month_match = re.search(
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            lower_message,
        )
        if ordinal_day_month_match and details["date"] is None:
            details["date"] = f"{ordinal_day_month_match.group(1)} {ordinal_day_month_match.group(2).title()}"

        day_month_year_match = re.search(r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b", lower_message)
        if day_month_year_match and details["date"] is None:
            details["date"] = f"{day_month_year_match.group(1)} {day_month_year_match.group(2).title()} {day_month_year_match.group(3)}"

        day_month_match = re.search(r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b", lower_message)
        if day_month_match and details["date"] is None:
            details["date"] = f"{day_month_match.group(1)} {day_month_match.group(2).title()}"

        # Extract time
        time_match = re.search(r"\b(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", lower_message)
        if time_match:
            details["time"] = time_match.group(1).upper()
        elif re.search(r"\b(\d{1,2})\b", lower_message):
            numeric_match = re.search(r"\b(\d{1,2})\b", lower_message)
            if numeric_match and numeric_match.group(1) not in {"2", "4", "6", "8"}:
                details["time"] = numeric_match.group(1)

        # Extract party size
        if "only me" in lower_message or "just me" in lower_message or "one person" in lower_message:
            details["party_size"] = "1"
        party_match = re.search(r"\b(?:party\s+of|for|for\s+)(\d+)\b(?:\s*(?:guests?|people))?", lower_message)
        if party_match:
            details["party_size"] = party_match.group(1)
        elif state and state.get("awaiting") == "party_size" and re.search(r"\b(\d+)\b", lower_message):
            numeric_match = re.search(r"\b(\d+)\b", lower_message)
            if numeric_match:
                details["party_size"] = numeric_match.group(1)

        return details

    def forward_to_staff(self, original_message: str, sender: str, reason: str):
        """
        Forward important messages to staff using WhatsApp if configured.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert = f"""
🚨 HUMAN INTERVENTION NEEDED 🚨
Time: {timestamp}
From: {sender}
Message: {original_message}
Reason: {reason}
"""
        print(alert)

        if not self.staff_phone_number or not self.meta_access_token or not self.meta_phone_number_id:
            return

        try:
            url = f"https://graph.facebook.com/{self.meta_api_version}/{self.meta_phone_number_id}/messages"
            headers = {
                "Authorization": f"Bearer {self.meta_access_token}",
                "Content-Type": "application/json"
            }
            payload = {
                "messaging_product": "whatsapp",
                "to": self.staff_phone_number,
                "type": "text",
                "text": {"body": alert.strip()},
            }
            response = httpx.post(url, headers=headers, json=payload, timeout=10.0)
            response.raise_for_status()
        except Exception as exc:
            print(f"Failed to notify staff via WhatsApp: {exc}")

    def update_business_info(self, category: str, data: dict):
        """
        Update business information dynamically
        """
        if category in self.business_info:
            self.business_info[category].update(data)
            return True
        return False

    def get_ai_provider_status(self) -> dict:
        """
        Get the status of all AI providers for debugging
        """
        return {
            "groq": {
                "available": self.groq_client is not None,
                "model": self.groq_model if self.groq_client else None
            },
            "openai": {
                "available": self.client is not None,
                "model": "gpt-3.5-turbo" if self.client else None
            },
            "huggingface": {
                "available": self.huggingface_api_token is not None and InferenceClient is not None,
                "model": self.huggingface_model if self.huggingface_api_token else None
            }
        }