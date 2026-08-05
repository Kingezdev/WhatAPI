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
        self.huggingface_client = None
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
                    {"name": "Pasta Carbonara", "price": "$18.99", "description": "Creamy pasta with bacon and parmesan"}
                ],
                "desserts": [
                    {"name": "Chocolate Cake", "price": "$7.99", "description": "Rich chocolate layer cake"},
                    {"name": "Ice Cream", "price": "$5.99", "description": "Choice of vanilla, chocolate, or strawberry"}
                ],
                "drinks": [
                    {"name": "Soft Drinks", "price": "$3.99", "description": "Coke, Sprite, Fanta"},
                    {"name": "Fresh Juice", "price": "$6.99", "description": "Orange, Apple, or Mango juice"},
                    {"name": "Coffee", "price": "$4.99", "description": "Espresso, Cappuccino, Latte"}
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
1. Answer questions about the menu, prices, opening hours, and location
2. Take reservations by collecting: name, date, time, and number of guests
3. Handle common customer inquiries professionally
4. Forward important or complex requests to human staff by indicating "HUMAN_NEEDED: [reason]"

Business Information:
{json.dumps(self.business_info, indent=2)}

Guidelines:
- Be friendly, professional, and concise
- For reservations, collect all required information
- If a request is complex, urgent, or requires human judgment, use HUMAN_NEEDED
- Keep responses under 160 characters when possible (WhatsApp limit)
- Use emojis to make responses friendly 🍽️📍⏰"""

        self.conversation_state = self._load_conversation_state()
        self.staff_phone_number = os.getenv("STAFF_WHATSAPP_NUMBER")
        self.meta_access_token = os.getenv("META_ACCESS_TOKEN")
        self.meta_phone_number_id = os.getenv("META_PHONE_NUMBER_ID")
        self.meta_api_version = os.getenv("META_API_VERSION", "v18.0")

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
        Generate AI response using providers in priority order:
        1. Groq (fastest, free tier)
        2. OpenAI (fallback)
        3. Hugging Face (final fallback)
        """
        sender = sender or "unknown"
        state = self.conversation_state.setdefault(
            sender,
            {
                "reservation_details": {"name": None, "date": None, "time": None, "party_size": None},
                "awaiting": None,
                "history": [],
            },
        )
        state["history"].append({"role": "user", "content": message})

        # 1. Try business info queries first (fast, no API call)
        business_response = self._handle_business_info_query(message, state)
        if business_response is not None:
            self._save_conversation_state(sender, state)
            state["history"].append({"role": "assistant", "content": business_response})
            return business_response

        # 2. Try reservation flow
        reservation_flow = self._handle_reservation_flow(message, state, sender)
        if reservation_flow is not None:
            self._save_conversation_state(sender, state)
            state["history"].append({"role": "assistant", "content": reservation_flow})
            return reservation_flow

        # 3. Try AI providers in priority order
        ai_response = None
        
        # Try Groq first (fastest, free)
        if self.groq_client and not ai_response:
            ai_response = self._generate_groq_response(message, state)
        
        # Try OpenAI as second choice
        if not ai_response and self.client:
            ai_response = self._generate_openai_response(message, state)
        
        # Try Hugging Face as final fallback
        if not ai_response:
            ai_response = self._generate_huggingface_response(message, sender)
        
        # If all AI providers fail, use fallback
        if not ai_response:
            ai_response = self._get_fallback_response(message)

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

            response = chat_completion.choices[0].message.content.strip()
            return response

        except Exception as e:
            print(f"Groq API error: {e}")
            return None

    def _generate_openai_response(self, message: str, state: dict) -> Optional[str]:
        """
        Generate response using OpenAI API (fallback from Groq)
        """
        try:
            if not self.client:
                return None

            history_messages = [{"role": "system", "content": self.system_prompt}]
            for item in state["history"][-6:]:
                history_messages.append({"role": item["role"], "content": item["content"]})

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=history_messages,
                max_tokens=150,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None

    def _is_supported_use_case(self, message: str) -> bool:
        lower_message = message.lower().strip()
        supported_topics = [
            "menu", "vegetarian", "veggie", "kids", "popular", "dish", "jollof", "open", "hours",
            "sunday", "located", "location", "address", "deliver", "delivery", "reservation", "book",
            "reserve", "table", "guest", "party", "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "thank", "thanks", "cancel", "nevermind", "how many", "status"
        ]
        if any(topic in lower_message for topic in supported_topics):
            return True

        if lower_message in {"", "?"}:
            return False

        return False

    def _generate_huggingface_response(self, message: str, sender: str) -> Optional[str]:
        """
        Generate response using Hugging Face API (final fallback)
        """
        if not self._is_supported_use_case(message):
            return "Please call for more enquiries."

        token = self.huggingface_api_token or os.getenv("HUGGINGFACE_API_TOKEN")
        if not token or InferenceClient is None:
            return None

        model = self.huggingface_model or os.getenv("HUGGINGFACE_MODEL", "google/flan-t5-base")
        prompt = (
            f"You are a helpful restaurant assistant for {self.business_info.get('name', 'this business')}. "
            f"Only answer questions related to the restaurant's menu, opening hours, location, delivery, or reservations. "
            f"If the message is unrelated, respond briefly with: Please call for more enquiries. "
            f"Customer message: {message}"
        )

        try:
            if self.huggingface_client is None:
                self.huggingface_client = InferenceClient(model=model, token=token)

            response = None
            try:
                response = self.huggingface_client.text2text_generation(
                    prompt,
                    max_new_tokens=80,
                    temperature=0.2,
                    do_sample=False,
                )
            except Exception as text2text_error:
                print(f"text2text_generation failed: {text2text_error}")
                try:
                    response = self.huggingface_client.text_generation(
                        prompt,
                        max_new_tokens=80,
                        temperature=0.2,
                        do_sample=False,
                    )
                except Exception as text_generation_error:
                    print(f"text_generation failed: {text_generation_error}")

            if isinstance(response, list):
                response = response[0].get("generated_text", "") if response and isinstance(response[0], dict) else ""
            if isinstance(response, str):
                return response.strip() or None
        except Exception as e:
            print(f"Error calling Hugging Face: {e}")

        return None

    def _get_fallback_response(self, message: str) -> str:
        """
        Return a fallback response when all AI providers fail
        """
        fallbacks = [
            "I'm here to help! 😊 Would you like to make a reservation, see our menu, or check our hours?",
            "How can I assist you today? I can help with reservations, menu info, opening hours, and more!",
            "Is there something specific you'd like to know about our restaurant? We're here to help!",
            "We have great menu options and comfortable seating! Would you like to know more or make a reservation?",
            "I can help with reservations, menu items, hours, and location. What would you like to know? 🍽️"
        ]
        import random
        return random.choice(fallbacks)

    def _handle_business_info_query(self, message: str, state: dict) -> str | None:
        lower_message = message.lower().strip()
        menu = self.business_info.get("menu", {})
        opening_hours = self.business_info.get("opening_hours", {})
        location = self.business_info.get("location", {})

        if any(phrase in lower_message for phrase in ["what's on your menu", "what is on your menu", "what is your menu", "what's your menu"]):
            items = []
            for section in ["appetizers", "main_courses", "desserts", "drinks", "kids_menu"]:
                for item in menu.get(section, []):
                    name = item.get("name")
                    if name:
                        items.append(name)
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

    def _handle_reservation_flow(self, message: str, state: dict, sender: str) -> str | None:
        """
        Handle reservation flow with proper confirmation, cancellation, and status checking.
        """
        lower_message = message.lower().strip()
        reservation_keywords = ["reservation", "book a table", "reserve", "table for", "book table"]
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

        # Handle cancellation requests first
        if any(word in lower_message for word in ["cancel", "abort", "stop", "nevermind", "forget it"]):
            if state["awaiting"] is not None:
                state["awaiting"] = None
                # Reset reservation details
                state["reservation_details"] = {"name": None, "date": None, "time": None, "party_size": None}
                self._save_conversation_state(sender, state)
                return "❌ Reservation cancelled. How else can I help you today? 😊"
            return None

        # Handle greeting
        if lower_message in greetings or lower_message.startswith(tuple(greetings)):
            state["awaiting"] = None
            self._save_conversation_state(sender, state)
            return "Hello! I can help you with reservations, opening hours, menu info, and location details. How can I assist you today?"

        # Handle reservation status query
        if any(keyword in lower_message for keyword in ["my reservation", "my booking", "reservation status", "check my reservation", "do i have a reservation", "how many reservation", "how many bookings"]):
            # Check for active reservations in the database
            active_reservations = self._get_active_reservations(sender)
            
            if active_reservations:
                count = len(active_reservations)
                if count == 1:
                    res = active_reservations[0]
                    return f"I found a reservation record for {res['name']} on {res['date']} at {res['time']} for {res['party_size']} guests. 📝"
                else:
                    # Multiple reservations
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
            
            return "I do not see any active reservations for you. Would you like to make one? 📝"

        # Check if we're in the middle of a reservation flow
        if state["awaiting"] is not None:
            return self._continue_reservation_flow(message, state, sender)

        # Start a new reservation if requested
        if any(keyword in lower_message for keyword in reservation_keywords):
            return self._start_reservation_flow(state, sender)

        return None

    def _start_reservation_flow(self, state: dict, sender: str) -> str:
        """
        Start a new reservation process.
        """
        state["awaiting"] = "name"
        self._save_conversation_state(sender, state)
        return "Absolutely — I can help with that. What name should I book under? 📝"

    def _continue_reservation_flow(self, message: str, state: dict, sender: str) -> str:
        """
        Continue an existing reservation flow based on what we're awaiting.
        """
        details = state["reservation_details"]
        
        # Extract any information from the message
        extracted = self._extract_reservation_details(message, state)
        for key, value in extracted.items():
            if value is not None:
                details[key] = value

        # Handle each step based on what we're waiting for
        if state["awaiting"] == "name":
            if details["name"] is None:
                return "What name should I book under? 📝"
            state["awaiting"] = "date"
            self._save_conversation_state(sender, state)
            return f"Great {details['name']}! What date would you like to reserve? 📅"

        if state["awaiting"] == "date":
            if details["date"] is None:
                return "What date would you like to reserve? 📅"
            state["awaiting"] = "time"
            self._save_conversation_state(sender, state)
            return f"Perfect! What time would you like to come in? ⏰"

        if state["awaiting"] == "time":
            if details["time"] is None:
                return "What time would you like to come in? ⏰"
            state["awaiting"] = "party_size"
            self._save_conversation_state(sender, state)
            return f"Wonderful! How many guests will be joining you? 🍽️"

        if state["awaiting"] == "party_size":
            if details["party_size"] is None:
                return "How many guests will be joining you? 🍽️"
            
            # All details collected - confirm and save
            state["awaiting"] = None
            self._save_reservation(sender, details)
            
            # Forward to staff
            self.forward_to_staff(
                f"Reservation: {details['name']} on {details['date']} at {details['time']} for {details['party_size']} guests",
                sender=sender,
                reason="New reservation created"
            )
            
            response = f"📝 Reservation confirmed for {details['name']} on {details['date']} at {details['time']} for {details['party_size']} guests! 🎉\n\nWe'll see you then! 😊"
            
            # Reset for next reservation
            state["reservation_details"] = {"name": None, "date": None, "time": None, "party_size": None}
            self._save_conversation_state(sender, state)
            
            return response

        return "I'm not sure what you mean. Let's start over. What name should I book under? 📝"

    def _save_reservation(self, sender: str, details: dict):
        """
        Save a reservation to the database.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO reservations (
                        sender, name, date, time, party_size, created_at, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sender,
                        details["name"],
                        details["date"],
                        details["time"],
                        details["party_size"],
                        datetime.now().isoformat(),
                        "active"
                    )
                )
                conn.commit()
        except Exception as e:
            print(f"Error saving reservation: {e}")

    def _get_active_reservations(self, sender: str) -> list:
        """
        Get all active reservations for a sender.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT name, date, time, party_size FROM reservations WHERE sender = ? AND status = 'active'",
                    (sender,)
                ).fetchall()
                
            return [
                {"name": row[0], "date": row[1], "time": row[2], "party_size": row[3]}
                for row in rows
            ]
        except Exception as e:
            print(f"Error retrieving reservations: {e}")
            return []

    def _get_saved_reservation(self, state: dict) -> dict | None:
        """
        Get a single saved reservation (for backward compatibility).
        """
        reservations = self._get_active_reservations("reservation")  # Default sender
        if reservations:
            return reservations[0]
        return None

    def _extract_reservation_details(self, message: str, state: dict | None = None) -> dict:
        details = {"name": None, "date": None, "time": None, "party_size": None}
        lower_message = message.lower()

        # Extract name
        if re.search(r"\bname\s+is\s+([a-z][a-z\s'-]+)", lower_message):
            match = re.search(r"\bname\s+is\s+([a-z][a-z\s'-]+)", lower_message)
            details["name"] = match.group(1).strip().title()
        elif re.search(r"for\s+([a-z][a-z\s'-]+)", lower_message) and "for 4" not in lower_message:
            match = re.search(r"for\s+([a-z][a-z\s'-]+)", lower_message)
            details["name"] = match.group(1).strip().title()
        elif state and state.get("awaiting") == "name" and re.fullmatch(r"[a-zA-Z][a-zA-Z\s'-]+", message.strip()):
            details["name"] = message.strip().title()

        # Extract date
        date_match = re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lower_message)
        if date_match:
            details["date"] = date_match.group(0).title()

        # Handle "today", "tomorrow"
        if "today" in lower_message:
            details["date"] = "today"
        elif "tomorrow" in lower_message:
            details["date"] = "tomorrow"

        day_month_year_match = re.search(r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b", lower_message)
        if day_month_year_match:
            details["date"] = f"{day_month_year_match.group(1)} {day_month_year_match.group(2).title()} {day_month_year_match.group(3)}"

        day_month_match = re.search(r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b", lower_message)
        if day_month_match:
            details["date"] = f"{day_month_match.group(1)} {day_month_match.group(2).title()}"

        # Extract time
        time_match = re.search(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", lower_message)
        if time_match:
            details["time"] = time_match.group(1).upper()
        elif re.search(r"\b(\d{1,2})\b", lower_message):
            numeric_match = re.search(r"\b(\d{1,2})\b", lower_message)
            if numeric_match and numeric_match.group(1) not in {"2", "4", "6", "8"}:
                details["time"] = numeric_match.group(1)

        # Extract party size
        party_match = re.search(r"\b(?:party of|for|for\s+)(\d+)\b", lower_message)
        if party_match:
            details["party_size"] = party_match.group(1)
        elif state and state.get("awaiting") == "party_size" and re.search(r"\b(\d+)\b", lower_message):
            numeric_match = re.search(r"\b(\d+)\b", lower_message)
            if numeric_match:
                details["party_size"] = numeric_match.group(1)

        if not details["time"] and re.search(r"\b(\d{1,2})\b", lower_message):
            numeric_match = re.search(r"\b(\d{1,2})\b", lower_message)
            if numeric_match and numeric_match.group(1) not in {"2", "4", "6", "8"}:
                details["time"] = numeric_match.group(1)

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