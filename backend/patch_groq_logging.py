import re

with open('app/services/claims.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_groq = '''        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()

        raw_text = data["choices"][0]["message"]["content"].strip()
        
        # Clean markdown fences if any'''

new_groq = '''        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            
            logger.info("Groq model configured: %s", model)
            logger.info("Groq HTTP status: %d", response.status_code)
            
            response.raise_for_status()
            
            try:
                data = response.json()
            except Exception as e:
                logger.error("Groq response is not valid JSON. Response text preview: %s", response.text[:300])
                raise
                
        has_choices = "choices" in data
        logger.info("Groq response contains 'choices': %s", has_choices)
        if has_choices:
            logger.info("Groq number of choices: %d", len(data["choices"]))
            if len(data["choices"]) > 0:
                choice = data["choices"][0]
                has_message = "message" in choice
                logger.info("Groq choice 0 contains 'message': %s", has_message)
                if has_message:
                    content_val = choice["message"].get("content")
                    logger.info("Groq message.content type: %s", type(content_val))
                    if content_val is not None:
                        logger.info("Groq message.content length: %d", len(str(content_val)))
                        logger.info("Groq message.content preview: %s", str(content_val)[:300])
                    else:
                        logger.warning("Groq message.content is None")

        raw_text = data["choices"][0]["message"]["content"] or ""
        raw_text = raw_text.strip()
        
        if not raw_text:
            logger.error("Groq raw_text is empty!")
            
        # Clean markdown fences if any'''

if old_groq in content:
    content = content.replace(old_groq, new_groq)
    with open('app/services/claims.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Patched claims.py successfully.")
else:
    print("Could not find the target block in claims.py.")
