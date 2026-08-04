import hashlib
import hmac
import json
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from ai_service import AIAssistant

load_dotenv()

app = FastAPI()

# Meta Cloud API Configuration
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
META_PHONE_NUMBER_ID = os.getenv("META_PHONE_NUMBER_ID")
META_API_VERSION = os.getenv("META_API_VERSION", "v18.0")
META_WEBHOOK_VERIFY_TOKEN = os.getenv("META_WEBHOOK_VERIFY_TOKEN")
META_WEBHOOK_SECRET = os.getenv("META_WEBHOOK_SECRET") or os.getenv("META_APP_SECRET")

# Initialize AI Assistant
ai_assistant = AIAssistant()


def verify_webhook_signature(payload: bytes, signature_header: str | None) -> bool:
    if not META_WEBHOOK_SECRET:
        return True
    if not signature_header:
        return False

    expected = hmac.new(
        META_WEBHOOK_SECRET.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(signature_header, f"sha256={expected}")


async def send_whatsapp_message(phone_number: str, message: str):
    """
    Send a message via Meta Cloud API.
    """
    if not META_ACCESS_TOKEN or not META_PHONE_NUMBER_ID:
        return {"status": "skipped", "reason": "missing Meta credentials"}

    url = f"https://graph.facebook.com/{META_API_VERSION}/{META_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {"body": message},
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data, timeout=10.0)
        return response.json()


@app.get("/")
async def root():
    return {"message": "WhatsApp AI Assistant API is running"}


@app.get("/webhook/whatsapp")
async def verify_webhook(request: Request):
    """
    Verify webhook with Meta Cloud API.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == META_WEBHOOK_VERIFY_TOKEN:
        return int(challenge)
    raise HTTPException(status_code=403, detail="Invalid verification token")


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    Webhook endpoint for Meta Cloud API WhatsApp messages.
    """
    payload = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not verify_webhook_signature(payload, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        data = json.loads(payload.decode("utf-8"))

        entry = data.get("entry", [])
        if not entry:
            return {"status": "ok"}

        changes = entry[0].get("changes", [])
        if not changes:
            return {"status": "ok"}

        value = changes[0].get("value", {})
        messages = value.get("messages", [])

        if not messages:
            return {"status": "ok"}

        message = messages[0]
        message_body = message.get("text", {}).get("body", "").strip()
        sender_number = message.get("from")

        if not message_body or not sender_number:
            return {"status": "ok"}

        ai_response = ai_assistant.generate_response(message_body, sender_number)
        await send_whatsapp_message(sender_number, ai_response)

        return {"status": "ok"}

    except Exception as exc:
        print(f"Error processing webhook: {exc}")
        return {"status": "error"}


@app.post("/webhook/whatsapp/test")
async def test_webhook(request: Request):
    """
    Test endpoint for manual testing.
    """
    try:
        data = await request.json()
        message = data.get("message", "")
        sender = data.get("sender", "1234567890")

        ai_response = ai_assistant.generate_response(message, sender)

        if data.get("send", False):
            await send_whatsapp_message(sender, ai_response)

        return JSONResponse(content={
            "message": message,
            "response": ai_response,
            "sender": sender
        })

    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
