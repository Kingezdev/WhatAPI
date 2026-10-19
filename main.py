import hashlib
import hmac
import json
import logging
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse

from ai_service import AIAssistant

load_dotenv()

# --- Logging -----------------------------------------------------------------
# Shows up in Render logs alongside uvicorn's own output.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     [%(name)s] %(message)s",
)
log = logging.getLogger("bot")

app = FastAPI()

# --- Meta Cloud API configuration --------------------------------------------
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
META_PHONE_NUMBER_ID = os.getenv("META_PHONE_NUMBER_ID")
META_API_VERSION = os.getenv("META_API_VERSION", "v18.0")
META_WEBHOOK_VERIFY_TOKEN = os.getenv("META_WEBHOOK_VERIFY_TOKEN")
META_WEBHOOK_SECRET = os.getenv("META_WEBHOOK_SECRET") or os.getenv("META_APP_SECRET")

# --- AI assistant ------------------------------------------------------------
ai_assistant = AIAssistant()

# --- Startup diagnostics (so misconfiguration is visible in the logs) ---------
log.info("AI provider status: %s", ai_assistant.get_ai_provider_status())
log.info(
    "Config check: META_ACCESS_TOKEN set=%s, META_PHONE_NUMBER_ID set=%s, "
    "VERIFY_TOKEN set=%s, SIGNATURE verification enabled=%s, API version=%s",
    bool(META_ACCESS_TOKEN),
    bool(META_PHONE_NUMBER_ID),
    bool(META_WEBHOOK_VERIFY_TOKEN),
    bool(META_WEBHOOK_SECRET),
    META_API_VERSION,
)


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


async def send_whatsapp_message(phone_number: str, message: str) -> bool:
    """
    Send a message via Meta Cloud API.
    Returns True on success, False on failure. Failures are always logged.
    """
    if not META_ACCESS_TOKEN or not META_PHONE_NUMBER_ID:
        log.error("Cannot send message: META_ACCESS_TOKEN or META_PHONE_NUMBER_ID is missing")
        return False

    url = f"https://graph.facebook.com/{META_API_VERSION}/{META_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {"body": message},
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data, timeout=15.0)
    except httpx.HTTPError:
        log.exception("Network error while sending WhatsApp message to %s", phone_number)
        return False

    if response.status_code >= 400:
        # A 401 / "Error validating access token" here means your Meta token expired.
        log.error("WhatsApp send FAILED (%s): %s", response.status_code, response.text)
        return False

    log.info("WhatsApp send OK to %s", phone_number)
    return True


@app.get("/")
async def root():
    return {"message": "WhatsApp AI Assistant API is running"}


@app.head("/")
async def root_head():
    # Keep-alive pingers (e.g. UptimeRobot) often use HEAD; avoid 405 noise.
    return PlainTextResponse("")


@app.get("/webhook/whatsapp")
async def verify_webhook(request: Request):
    """
    Verify webhook with Meta Cloud API.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == META_WEBHOOK_VERIFY_TOKEN and challenge:
        log.info("Webhook verified by Meta")
        return PlainTextResponse(challenge)

    log.warning("Webhook verification failed (mode=%s)", mode)
    raise HTTPException(status_code=403, detail="Invalid verification token")


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    Webhook endpoint for Meta Cloud API WhatsApp messages.
    """
    payload = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    if not verify_webhook_signature(payload, signature):
        log.warning("Rejected webhook: invalid or missing signature")
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        data = json.loads(payload.decode("utf-8"))

        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])

                # Delivery / read receipts arrive as "statuses" - nothing to answer.
                if not messages:
                    if value.get("statuses"):
                        log.debug("Ignoring status update")
                    continue

                for message in messages:
                    sender_number = message.get("from")
                    message_type = message.get("type")
                    message_body = (message.get("text") or {}).get("body", "").strip()

                    if not sender_number or not message_body:
                        log.info(
                            "Ignoring non-text message (type=%s) from %s",
                            message_type,
                            sender_number,
                        )
                        continue

                    log.info("Incoming message from %s: %s", sender_number, message_body)

                    # generate_response is synchronous (blocking Groq calls + SQLite),
                    # so run it in a worker thread to keep the event loop responsive.
                    ai_response = await run_in_threadpool(
                        ai_assistant.generate_response, message_body, sender_number
                    )
                    log.info("Reply to %s: %s", sender_number, ai_response)

                    await send_whatsapp_message(sender_number, ai_response)

        return {"status": "ok"}

    except Exception:
        # Still return 200 so Meta doesn't retry endlessly, but log the full traceback.
        log.exception("Error processing webhook")
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

        ai_response = await run_in_threadpool(ai_assistant.generate_response, message, sender)

        sent = None
        if data.get("send", False):
            sent = await send_whatsapp_message(sender, ai_response)

        return JSONResponse(
            content={
                "message": message,
                "response": ai_response,
                "sender": sender,
                "sent": sent,
            }
        )

    except Exception as exc:
        log.exception("Error in test webhook")
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.get("/debug/status")
async def debug_status():
    """
    Quick health check of providers/config. Remove or protect this before going to production.
    """
    return {
        "ai": ai_assistant.get_ai_provider_status(),
        "meta_token_set": bool(META_ACCESS_TOKEN),
        "meta_phone_number_id_set": bool(META_PHONE_NUMBER_ID),
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)