import httpx
import asyncio
from app.core.config import settings

async def list_models():
    api_key = settings.GROQ_API_KEY.get_secret_value()
    async with httpx.AsyncClient() as client:
        res = await client.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {api_key}"})
        for m in res.json().get("data", []):
            print(m["id"])

asyncio.run(list_models())
