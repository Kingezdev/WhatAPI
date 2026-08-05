import json
import os
import re
import sqlite3
from datetime import datetime, time

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class AIAssistant:
    def __init__(self, db_path: str | None = None):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
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
        Generate AI response to incoming message while keeping per-sender reservation context.
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

        business_response = self._handle_business_info_query(message, state)
        if business_response is not None:
            self._save_conversation_state(sender, state)
            state["history"].append({"role": "assistant", "content": business_response})
            return business_response

        reservation_flow = self._handle_reservation_flow(message, state)
        if reservation_flow is not None:
            self._save_conversation_state(sender, state)
            state["history"].append({"role": "assistant", "content": reservation_flow})
            return reservation_flow

        try:
            if not self.client:
                return "I can help with the menu, hours, and reservations. What would you like to know?"

            history_messages = [{"role": "system", "content": self.system_prompt}]
            for item in state["history"][-6:]:
                history_messages.append({"role": item["role"], "content": item["content"]})

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=history_messages,
                max_tokens=150,
                temperature=0.7
            )

            ai_response = response.choices[0].message.content.strip()

            state["history"].append({"role": "assistant", "content": ai_response})
            self._save_conversation_state(sender, state)

            if "HUMAN_NEEDED:" in ai_response:
                self.forward_to_staff(message, sender, ai_response)
                return "Your request has been forwarded to our team. They'll get back to you shortly! 👨‍💼"

            return ai_response

        except Exception as e:
            print(f"Error generating AI response: {e}")
            return "Sorry, I'm having trouble right now. Please try again later or call us directly."

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

    def _handle_reservation_flow(self, message: str, state: dict) -> str | None:
        lower_message = message.lower().strip()
        reservation_keywords = ["reservation", "book a table", "reserve", "table for", "book table"]
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

        if lower_message in greetings or lower_message.startswith(tuple(greetings)):
            state["awaiting"] = None
            return "Hello! I can help you with reservations, opening hours, menu info, and location details. How can I assist you today?"

        if any(keyword in lower_message for keyword in ["my reservation", "my booking", "reservation status", "check my reservation", "do i have a reservation"]):
            details = state["reservation_details"]
            if any(details.get(key) for key in ["name", "date", "time", "party_size"]):
                name = details.get("name") or "your name"
                date = details.get("date") or "a selected date"
                time = details.get("time") or "a selected time"
                party_size = details.get("party_size") or "a selected party size"
                return f"I found a reservation record for {name} on {date} at {time} for {party_size} guests."
            return "I do not see a reservation record for you yet."

        if state["awaiting"] is None and not any(keyword in lower_message for keyword in reservation_keywords):
            return None

        details = state["reservation_details"]
        extracted = self._extract_reservation_details(message, state)
        for key, value in extracted.items():
            if value is not None:
                details[key] = value

        if details["name"] is None:
            state["awaiting"] = "name"
            return "Absolutely — I can help with that. What name should I book under?"

        if details["date"] is None:
            state["awaiting"] = "date"
            return "Great, what date would you like to reserve?"

        if details["time"] is None:
            state["awaiting"] = "time"
            return "Perfect. What time would you like to come in?"

        if details["party_size"] is None:
            state["awaiting"] = "party_size"
            return "Wonderful. How many guests will be joining you?"

        state["awaiting"] = None
        self.forward_to_staff(message, sender="reservation", reason="reservation completed")
        return "Thanks! I have your reservation details."

    def _extract_reservation_details(self, message: str, state: dict | None = None) -> dict:
        details = {"name": None, "date": None, "time": None, "party_size": None}
        lower_message = message.lower()

        if re.search(r"\bname\s+is\s+([a-z][a-z\s'-]+)", lower_message):
            match = re.search(r"\bname\s+is\s+([a-z][a-z\s'-]+)", lower_message)
            details["name"] = match.group(1).strip().title()
        elif re.search(r"for\s+([a-z][a-z\s'-]+)", lower_message) and "for 4" not in lower_message:
            match = re.search(r"for\s+([a-z][a-z\s'-]+)", lower_message)
            details["name"] = match.group(1).strip().title()
        elif state and state.get("awaiting") == "name" and re.fullmatch(r"[a-zA-Z][a-zA-Z\s'-]+", message.strip()):
            details["name"] = message.strip().title()

        date_match = re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lower_message)
        if date_match:
            details["date"] = date_match.group(0).title()

        day_month_year_match = re.search(r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b", lower_message)
        if day_month_year_match:
            details["date"] = f"{day_month_year_match.group(1)} {day_month_year_match.group(2).title()} {day_month_year_match.group(3)}"

        day_month_match = re.search(r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b", lower_message)
        if day_month_match:
            details["date"] = f"{day_month_match.group(1)} {day_month_match.group(2).title()}"

        time_match = re.search(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", lower_message)
        if time_match:
            details["time"] = time_match.group(1).upper()
        elif re.search(r"\b(\d{1,2})\b", lower_message):
            numeric_match = re.search(r"\b(\d{1,2})\b", lower_message)
            if numeric_match and numeric_match.group(1) not in {"2", "4", "6", "8"}:
                details["time"] = numeric_match.group(1)

        party_match = re.search(r"\b(?:party of|for|for\s+)(\d+)\b", lower_message)
        if party_match:
            details["party_size"] = party_match.group(1)
        elif state.get("awaiting") == "party_size" and re.search(r"\b(\d+)\b", lower_message):
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
