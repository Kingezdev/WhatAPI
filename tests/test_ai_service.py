import hashlib
import hmac
import os
import sys
from datetime import datetime
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main
from ai_service import AIAssistant


def test_reservation_context_is_retained_per_sender(monkeypatch):
    assistant = AIAssistant.__new__(AIAssistant)
    assistant.client = Mock()
    assistant.business_info = {
        "name": "Test Restaurant",
        "reservation_policy": "Reservations can be made for parties of 2 or more."
    }
    assistant.system_prompt = "system"
    assistant.conversation_state = {}
    assistant.forward_to_staff = Mock()

    first = assistant.generate_response("book a table", "+15551234567")
    second = assistant.generate_response("Alex", "+15551234567")
    third = assistant.generate_response("Friday", "+15551234567")
    fourth = assistant.generate_response("6pm", "+15551234567")
    fifth = assistant.generate_response("4", "+15551234567")

    assert first == "Absolutely — I can help with that. What name should I book under?"
    assert second == "Great, what date would you like to reserve?"
    assert third == "Perfect. What time would you like to come in?"
    assert fourth == "Wonderful. How many guests will be joining you?"
    assert fifth == "Thanks! I have your reservation details."
    assert assistant.conversation_state["+15551234567"]["reservation_details"] == {
        "name": "Alex",
        "date": "Friday",
        "time": "6PM",
        "party_size": "4"
    }


def test_webhook_signature_verification(monkeypatch):
    monkeypatch.setattr(main, "META_WEBHOOK_SECRET", "super-secret")
    payload = b"hello webhook"

    assert main.verify_webhook_signature(payload, None) is False

    expected = "sha256=" + hmac.new(
        b"super-secret",
        payload,
        hashlib.sha256,
    ).hexdigest()
    assert main.verify_webhook_signature(payload, expected) is True

    invalid = "sha256=" + hashlib.sha256(b"different").hexdigest()
    assert main.verify_webhook_signature(payload, invalid) is False


def test_forward_to_staff_sends_whatsapp_notification(monkeypatch):
    assistant = AIAssistant.__new__(AIAssistant)
    assistant.staff_phone_number = "+15551234567"
    assistant.meta_access_token = "token"
    assistant.meta_phone_number_id = "123"
    assistant.meta_api_version = "v18.0"

    post = Mock()
    post.return_value.raise_for_status.return_value = None
    monkeypatch.setattr("ai_service.httpx.post", post)

    assistant.forward_to_staff("help", "+15551234567", "human review")

    assert post.called


def test_reservation_status_reads_persisted_details(tmp_path):
    db_path = tmp_path / "conversations.sqlite3"

    assistant_one = AIAssistant(db_path=str(db_path))
    assistant_one.client = Mock()
    assistant_one.business_info = {
        "name": "Test Restaurant",
        "reservation_policy": "Reservations can be made for parties of 2 or more."
    }
    assistant_one.system_prompt = "system"
    assistant_one.forward_to_staff = Mock()

    assistant_one.generate_response("book a table", "+15551234567")
    assistant_one.generate_response("Alex", "+15551234567")
    assistant_one.generate_response("Friday", "+15551234567")
    assistant_one.generate_response("6pm", "+15551234567")
    assistant_one.generate_response("4", "+15551234567")

    assistant_two = AIAssistant(db_path=str(db_path))
    assistant_two.client = Mock()
    assistant_two.business_info = {
        "name": "Test Restaurant",
        "reservation_policy": "Reservations can be made for parties of 2 or more."
    }
    assistant_two.system_prompt = "system"
    assistant_two.forward_to_staff = Mock()

    response = assistant_two.generate_response("check my reservation", "+15551234567")

    assert response == "I found a reservation record for Alex on Friday at 6PM for 4 guests."


def test_rule_based_business_questions(monkeypatch):
    assistant = AIAssistant.__new__(AIAssistant)
    assistant.client = Mock()
    assistant.business_info = {
        "name": "Test Restaurant",
        "description": "A cozy neighborhood restaurant",
        "menu": {
            "appetizers": [{"name": "Spring Rolls", "price": "$8.99"}],
            "main_courses": [
                {"name": "Grilled Salmon", "price": "$24.99"},
                {"name": "Jollof Rice", "price": "$14.99"},
            ],
            "kids_menu": [{"name": "Kids Meal", "price": "$8.99"}],
        },
        "opening_hours": {
            "monday": "9:00 AM - 10:00 PM",
            "tuesday": "9:00 AM - 10:00 PM",
            "wednesday": "9:00 AM - 10:00 PM",
            "thursday": "9:00 AM - 10:00 PM",
            "friday": "9:00 AM - 11:00 PM",
            "saturday": "10:00 AM - 11:00 PM",
            "sunday": "10:00 AM - 9:00 PM",
        },
        "location": {"address": "123 Main Street, City, State 12345"},
        "reservation_policy": "Reservations can be made for parties of 2 or more."
    }
    assistant.system_prompt = "system"
    assistant.conversation_state = {}
    assistant.forward_to_staff = Mock()

    menu_response = assistant.generate_response("What's on your menu?", "+15551234567")
    vegetarian_response = assistant.generate_response("Do you have any vegetarian options?", "+15551234567")
    price_response = assistant.generate_response("How much is your jollof rice?", "+15551234567")
    kids_response = assistant.generate_response("Do you have a kids menu?", "+15551234567")
    popular_response = assistant.generate_response("What's your most popular dish?", "+15551234567")
    open_response = assistant.generate_response("What time do you open?", "+15551234567")
    sunday_response = assistant.generate_response("Are you open on Sundays?", "+15551234567")
    location_response = assistant.generate_response("Where are you located?", "+15551234567")
    deliver_response = assistant.generate_response("Do you deliver to Lekki?", "+15551234567")

    class FixedDateTime(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 8, 5, 18, 0, 0)

    monkeypatch.setattr("ai_service.datetime", FixedDateTime)
    open_now_response = assistant.generate_response("Are you open right now?", "+15551234567")

    assert "Spring Rolls" in menu_response and "Grilled Salmon" in menu_response
    assert "vegetarian" in vegetarian_response.lower()
    assert "$14.99" in price_response
    assert "Kids Meal" in kids_response and "$8.99" in kids_response
    assert "Jollof Rice" in popular_response
    assert "9:00 AM" in open_response
    assert "Sunday" in sunday_response and "10:00 AM" in sunday_response
    assert "123 Main Street" in location_response
    assert "deliver" in deliver_response.lower()
    assert "Yes" in open_now_response and "10:00 PM" in open_now_response


def test_huggingface_fallback_answers_unknown_messages(monkeypatch):
    assistant = AIAssistant.__new__(AIAssistant)
    assistant.client = None
    assistant.business_info = {
        "name": "Test Restaurant",
        "reservation_policy": "Reservations can be made for parties of 2 or more."
    }
    assistant.system_prompt = "system"
    assistant.conversation_state = {}
    assistant.forward_to_staff = Mock()
    assistant.huggingface_api_token = "fake-token"
    assistant.huggingface_model = "test-model"

    post = Mock()
    post.return_value.raise_for_status.return_value = None
    post.return_value.json.return_value = [{"generated_text": "A helpful fallback reply."}]
    monkeypatch.setattr("ai_service.httpx.post", post)

    response = assistant.generate_response("Can you help me with a refund?", "+15551234567")

    assert response == "Please call for more enquiries."
    assert not post.called


def test_reservation_state_persists_in_sqlite(tmp_path):
    db_path = tmp_path / "conversations.sqlite3"

    assistant_one = AIAssistant(db_path=str(db_path))
    assistant_one.client = Mock()
    assistant_one.business_info = {
        "name": "Test Restaurant",
        "reservation_policy": "Reservations can be made for parties of 2 or more."
    }
    assistant_one.system_prompt = "system"
    assistant_one.forward_to_staff = Mock()

    assistant_one.generate_response("book a table", "+15551234567")
    assistant_one.generate_response("Alex", "+15551234567")

    assistant_two = AIAssistant(db_path=str(db_path))
    assistant_two.client = Mock()
    assistant_two.business_info = {
        "name": "Test Restaurant",
        "reservation_policy": "Reservations can be made for parties of 2 or more."
    }
    assistant_two.system_prompt = "system"
    assistant_two.forward_to_staff = Mock()

    response = assistant_two.generate_response("Friday", "+15551234567")

    assert response == "Perfect. What time would you like to come in?"
    assert assistant_two.conversation_state["+15551234567"]["reservation_details"]["name"] == "Alex"
