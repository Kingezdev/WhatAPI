import os
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

token = os.getenv('HUGGINGFACE_API_TOKEN')
model = os.getenv('HUGGINGFACE_MODEL', 'google/flan-t5-base')
print('token present:', bool(token))
print('model:', model)

payloads = [
    {'inputs': 'Say hello'},
    {'inputs': 'Translate to French: Hello'},
    {'inputs': 'Answer briefly: What is 2+2?'}
]

for payload in payloads:
    print('payload', payload)
    req = urllib.request.Request(
        f'https://api-inference.huggingface.co/models/{model}',
        data=json.dumps(payload).encode(),
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            print('status', resp.status)
            print(resp.read().decode())
    except urllib.error.HTTPError as e:
        print('http_error', e.code)
        print(e.read().decode())
    except Exception as e:
        print('error', type(e).__name__, e)
