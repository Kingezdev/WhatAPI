import hashlib
import hmac
import os
import sys
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
