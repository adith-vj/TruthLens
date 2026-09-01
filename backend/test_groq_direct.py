import asyncio
import os
import json
import httpx
from app.core.config import settings
from app.services.claims import _EXTRACT_PROMPT

async def test_groq():
    api_key = settings.GROQ_API_KEY.get_secret_value()
    model = settings.GROQ_MODEL
    transcript_text = "This is a test transcript. The Eiffel Tower is 330 meters tall."
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": _EXTRACT_PROMPT.format(transcript_text=transcript_text)
            }
        ],
        "temperature": 0.0,
    }
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            print("Groq model configured:", model)
            print("Groq HTTP status:", response.status_code)
            
            try:
                data = response.json()
            except Exception as e:
                print("Groq response is not valid JSON. Response text preview:", response.text[:300])
                return

        has_choices = "choices" in data
        print("Groq response contains 'choices':", has_choices)
        if has_choices:
            print("Groq number of choices:", len(data["choices"]))
            if len(data["choices"]) > 0:
                choice = data["choices"][0]
                has_message = "message" in choice
                print("Groq choice 0 contains 'message':", has_message)
                if has_message:
                    content_val = choice["message"].get("content")
                    print("Groq message.content type:", type(content_val))
                    if content_val is not None:
                        print("Groq message.content length:", len(str(content_val)))
                        print("Groq message.content preview:", repr(str(content_val)[:300]))
                    else:
                        print("Groq message.content is None")
                        
    except Exception as exc:
        print("Exception:", exc)

if __name__ == '__main__':
    asyncio.run(test_groq())
