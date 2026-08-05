# test_ai.py
from ai_service import AIAssistant

assistant = AIAssistant()

# Check which providers are available
print("AI Provider Status:", assistant.get_ai_provider_status())

# Test messages
messages = [
    "What's on your menu?",
    "I'd like to book a table for 4",
    "Where are you located?",
    "Do you deliver?"
]

for msg in messages:
    print(f"\nUser: {msg}")
    response = assistant.generate_response(msg, "test_user")
    print(f"Assistant: {response}")