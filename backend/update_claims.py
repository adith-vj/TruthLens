import re

with open('app/services/claims.py', 'r', encoding='utf-8') as f:
    content = f.read()

# We need to add ClaimExtractionError
if 'class ClaimExtractionError' not in content:
    content = content.replace('from typing import Any', 'from typing import Any\n\nclass ClaimExtractionError(Exception):\n    \"\"\"Raised when claim extraction fails (e.g. rate limit, malformed JSON).\"\"\"\n    pass')

# Update _call_gemini to raise ClaimExtractionError instead of returning []
old_gemini = '''        if not isinstance(parsed, list):
            logger.error("Gemini returned non-list JSON for claims: %s", type(parsed))
            return []

        return parsed

    except httpx.HTTPStatusError as exc:
        logger.error(
            "Gemini claim extraction HTTP %d: %s",
            exc.response.status_code,
            exc.response.text[:300],
        )
        return []
    except Exception as exc:
        logger.error("Gemini claim extraction failed: %s", exc)
        return []'''

new_gemini = '''        if not isinstance(parsed, list):
            raise ClaimExtractionError(f"Expected list, got {type(parsed)}")

        return parsed

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        logger.error(
            "Gemini claim extraction HTTP %d: %s",
            status,
            exc.response.text[:300],
        )
        if status == 429:
            raise ClaimExtractionError("Gemini quota exceeded (HTTP 429)") from exc
        raise ClaimExtractionError(f"Gemini HTTP error {status}") from exc
    except httpx.TimeoutException as exc:
        logger.error("Gemini claim extraction timed out.")
        raise ClaimExtractionError("Gemini timeout") from exc
    except Exception as exc:
        if isinstance(exc, ClaimExtractionError):
            raise
        logger.error("Gemini claim extraction failed: %s", exc)
        raise ClaimExtractionError(f"Gemini failed: {exc}") from exc'''

if old_gemini in content:
    content = content.replace(old_gemini, new_gemini)
else:
    print("Could not find old_gemini block to replace.")

# Add _call_groq
groq_func = '''
# ---------------------------------------------------------------------------
# Groq fallback request
# ---------------------------------------------------------------------------

async def _call_groq(transcript_text: str, api_key: str, model: str) -> list[dict[str, Any]]:
    \"\"\"
    Send a request to Groq's OpenAI-compatible endpoint as a fallback.
    \"\"\"
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
            response.raise_for_status()
            data = response.json()

        raw_text = data["choices"][0]["message"]["content"].strip()
        
        # Clean markdown fences if any
        if raw_text.startswith("`"):
            lines = raw_text.splitlines()
            if lines[0].startswith("`"):
                lines = lines[1:]
            if lines and lines[-1].startswith("`"):
                lines = lines[:-1]
            raw_text = "\\n".join(lines).strip()
            
        parsed = json.loads(raw_text)

        if not isinstance(parsed, list):
            raise ClaimExtractionError(f"Groq expected list, got {type(parsed)}")

        return parsed

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        logger.error(
            "Groq fallback extraction HTTP %d: %s",
            status,
            exc.response.text[:300],
        )
        raise ClaimExtractionError(f"Groq HTTP error {status}") from exc
    except httpx.TimeoutException as exc:
        logger.error("Groq fallback extraction timed out.")
        raise ClaimExtractionError("Groq timeout") from exc
    except Exception as exc:
        if isinstance(exc, ClaimExtractionError):
            raise
        logger.error("Groq fallback extraction failed: %s", exc)
        raise ClaimExtractionError(f"Groq failed: {exc}") from exc
'''
if "_call_groq" not in content:
    content = content.replace("# Scoring helpers (pure Python, zero I/O)", groq_func + "\n# ---------------------------------------------------------------------------\n# Scoring helpers (pure Python, zero I/O)")


# Update process_video_claims loop
old_loop = '''    # 3. Send batches sequentially (avoids parallel 429s on free tier)
    raw_claims: list[dict[str, Any]] = []
    for i, batch_text in enumerate(batches):
        logger.info("Sending batch %d/%d to Gemini", i + 1, len(batches))
        results = await _call_gemini(batch_text, api_key)
        raw_claims.extend(results)'''

new_loop = '''    # 3. Send batches sequentially (avoids parallel 429s on free tier)
    raw_claims: list[dict[str, Any]] = []
    groq_api_key = settings.GROQ_API_KEY.get_secret_value()
    groq_model = settings.GROQ_MODEL
    
    for i, batch_text in enumerate(batches):
        logger.info("Sending batch %d/%d to Gemini", i + 1, len(batches))
        try:
            results = await _call_gemini(batch_text, api_key)
            if results:
                raw_claims.extend(results)
            else:
                logger.info("Gemini succeeded but returned no claims.")
        except ClaimExtractionError as exc:
            logger.warning("Gemini claim extraction failed with %s; attempting Groq fallback", exc)
            if groq_api_key:
                try:
                    results = await _call_groq(batch_text, groq_api_key, groq_model)
                    logger.info("Groq fallback claim extraction succeeded: %d claims extracted", len(results))
                    if results:
                        raw_claims.extend(results)
                except ClaimExtractionError as g_exc:
                    logger.error("Groq fallback claim extraction failed: %s", g_exc)
            else:
                logger.warning("GROQ_API_KEY not configured, skipping fallback.")'''

if old_loop in content:
    content = content.replace(old_loop, new_loop)
else:
    print("Could not find old_loop block to replace.")

with open('app/services/claims.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated app/services/claims.py")
