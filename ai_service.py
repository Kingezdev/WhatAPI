import os
import re
import json
from datetime import datetime

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class AIAssistant:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
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

        self.conversation_state = {}
        self.staff_phone_number = os.getenv("STAFF_WHATSAPP_NUMBER")
        self.meta_access_token = os.getenv("META_ACCESS_TOKEN")
        self.meta_phone_number_id = os.getenv("META_PHONE_NUMBER_ID")
        self.meta_api_version = os.getenv("META_API_VERSION", "v18.0")

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

        reservation_flow = self._handle_reservation_flow(message, state)
        if reservation_flow is not None:
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

            if "HUMAN_NEEDED:" in ai_response:
                self.forward_to_staff(message, sender, ai_response)
                return "Your request has been forwarded to our team. They'll get back to you shortly! 👨‍💼"

            return ai_response

        except Exception as e:
            print(f"Error generating AI response: {e}")
            return "Sorry, I'm having trouble right now. Please try again later or call us directly."

    def _handle_reservation_flow(self, message: str, state: dict) -> str | None:
        lower_message = message.lower().strip()
        reservation_keywords = ["reservation", "book a table", "reserve", "table for", "book table"]
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
