# WhatsApp AI Assistant for Business

A 24/7 AI-powered WhatsApp assistant that handles customer inquiries, takes reservations, answers questions about menu/prices/hours/location, and forwards important requests to staff.

## Features

- 🤖 **AI-Powered Responses**: Uses OpenAI GPT-3.5 for intelligent conversations
- 📅 **Reservation Handling**: Collects booking information automatically
- 📋 **Menu & Pricing**: Provides instant access to menu items and prices
- ⏰ **Opening Hours**: Answers questions about business hours
- 📍 **Location Info**: Shares address and contact details
- 👨‍💼 **Staff Escalation**: Forwards complex requests to human staff
- 🔄 **24/7 Availability**: Always online to help customers

## Prerequisites

- Python 3.8+
- Meta Business Account (with WhatsApp Business API access)
- OpenAI API Key

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your actual values:
- `META_ACCESS_TOKEN`: Get from Meta Business Suite > WhatsApp > API Setup
- `META_PHONE_NUMBER_ID`: Get from Meta Business Suite > WhatsApp > Phone Numbers
- `META_API_VERSION`: Meta Graph API version (default: v18.0)
- `META_WEBHOOK_VERIFY_TOKEN`: A secret token you create for webhook verification
- `OPENAI_API_KEY`: Get from OpenAI Platform

### 3. Set Up Meta Cloud API for WhatsApp

1. Go to [Meta Business Suite](https://business.facebook.com)
2. Navigate to WhatsApp > API Setup
3. Create or select your WhatsApp Business Account
4. Add a phone number and verify it
5. Generate an Access Token with necessary permissions
6. Note your Phone Number ID for the `.env` file
7. Choose a webhook verify token (any random string you keep secret)

### 4. Customize Business Information

Edit `ai_service.py` to update:
- Business name and description
- Menu items and prices
- Opening hours
- Location and contact info
- Reservation policies

### 5. Run the Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 6. Set Up Webhook (Production)

1. Deploy your server (Heroku, Railway, AWS, etc.)
2. Go to Meta Business Suite > WhatsApp > API Setup > Webhooks
3. Set the webhook URL to: `https://your-domain.com/webhook/whatsapp`
4. Set the verify token to match your `META_WEBHOOK_VERIFY_TOKEN`
5. Subscribe to webhook fields: `messages`
6. Ensure your server is publicly accessible

## Testing

### Local Testing with ngrok

1. Install ngrok: https://ngrok.com/download
2. Run ngrok: `ngrok http 8000`
3. Use the ngrok URL in Meta webhook settings
4. Send messages from WhatsApp to test

### Manual API Testing

```bash
curl -X POST http://localhost:8000/webhook/whatsapp/test \
  -H "Content-Type: application/json" \
  -d '{"message": "What are your opening hours?", "sender": "1234567890"}'
```

## Project Structure

```
WhatsappAPi/
├── main.py              # FastAPI application with webhook endpoints
├── ai_service.py        # AI assistant logic and business info
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
├── .env                # Your actual credentials (not in git)
└── README.md           # This file
```

## API Endpoints

- `GET /` - Health check endpoint
- `GET /webhook/whatsapp` - Meta webhook verification endpoint
- `POST /webhook/whatsapp` - Meta webhook endpoint for receiving messages
- `POST /webhook/whatsapp/test` - Manual testing endpoint

## Customization

### Adding New Menu Items

Edit the `menu` dictionary in `ai_service.py`:

```python
"menu": {
    "category_name": [
        {"name": "Item Name", "price": "$0.00", "description": "Description"}
    ]
}
```

### Changing AI Behavior

Modify the `system_prompt` in `ai_service.py` to adjust the AI's personality and response style.

### Adding Staff Notifications

Implement the `forward_to_staff` method in `ai_service.py` to send actual notifications via:
- Email (SendGrid, AWS SES)
- WhatsApp (Meta Cloud API)
- Slack/Discord webhooks
- Database storage for staff dashboard

## Security Notes

- Never commit `.env` to version control
- Use environment variables for all sensitive data
- Implement rate limiting for production
- Add authentication for admin endpoints
- Validate all incoming webhook requests

## Troubleshooting

### Meta Webhook Not Working
- Ensure ngrok/server is publicly accessible
- Check webhook URL is correct in Meta Business Suite
- Verify server is running on correct port
- Confirm webhook verify token matches in both places
- Check that you're subscribed to the `messages` field

### AI Not Responding
- Check OpenAI API key is valid
- Verify you have API credits available
- Check server logs for errors

### Messages Too Long
- AI responses are limited to 150 tokens
- Adjust `max_tokens` in `ai_service.py` if needed
- WhatsApp has 160-character limit for optimal display

## License

MIT License - feel free to use and modify for your business.
