import re

with open('app/services/claims.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix _call_groq empty content handling
old_groq_content = '''        if has_message:
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

new_groq_content = '''        if has_message:
                    content_val = choice["message"].get("content")
                    if content_val is not None:
                        logger.debug("Groq message.content length: %d", len(str(content_val)))

        raw_text = data["choices"][0]["message"].get("content") or ""
        raw_text = raw_text.strip()
        
        if not raw_text:
            logger.error("Groq returned empty/whitespace-only content")
            raise ClaimExtractionError("Groq returned empty content")
            
        # Clean markdown fences if any'''

content = content.replace(old_groq_content, new_groq_content)

# Remove the old temporary safe diagnostic logging that was too noisy
content = content.replace('logger.info("Groq model configured:', 'logger.debug("Groq model configured:')
content = content.replace('logger.info("Groq HTTP status:', 'logger.debug("Groq HTTP status:')
content = content.replace("logger.info(\"Groq response contains 'choices':", "logger.debug(\"Groq response contains 'choices':")
content = content.replace('logger.info("Groq number of choices:', 'logger.debug("Groq number of choices:')
content = content.replace("logger.info(\"Groq choice 0 contains 'message':", "logger.debug(\"Groq choice 0 contains 'message':")
content = content.replace('logger.error("Groq model configured:', 'logger.debug("Groq model configured:')
content = content.replace('logger.error("Groq HTTP status:', 'logger.debug("Groq HTTP status:')
content = content.replace("logger.error(\"Groq response contains 'choices':", "logger.debug(\"Groq response contains 'choices':")
content = content.replace('logger.error("Groq number of choices:', 'logger.debug("Groq number of choices:')
content = content.replace("logger.error(\"Groq choice 0 contains 'message':", "logger.debug(\"Groq choice 0 contains 'message':")


# Add the chunking function right above _call_groq
chunk_func = '''
def _chunk_text_for_groq(text: str, max_chars: int) -> list[str]:
    \"\"\"
    Splits a large transcript batch into smaller sub-batches for Groq.
    Preserves existing timestamps by splitting only on line breaks.
    \"\"\"
    lines = text.split("\\n")
    batches = []
    current_lines = []
    current_len = 0
    
    for line in lines:
        line_len = len(line) + 1  # +1 for newline
        if current_lines and current_len + line_len > max_chars:
            batches.append("\\n".join(current_lines))
            current_lines = []
            current_len = 0
        current_lines.append(line)
        current_len += line_len
        
    if current_lines:
        batches.append("\\n".join(current_lines))
        
    return batches

'''

if '_chunk_text_for_groq' not in content:
    content = content.replace('async def _call_groq', chunk_func + 'async def _call_groq')


# Update the loop in process_video_claims
old_loop = '''    for i, batch_text in enumerate(batches):
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

new_loop = '''    for i, batch_text in enumerate(batches):
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
                # Target max input characters based on tokens (4 chars/token approximation)
                groq_max_chars = settings.GROQ_MAX_INPUT_TOKENS * 4
                groq_sub_batches = _chunk_text_for_groq(batch_text, groq_max_chars)
                logger.info("Groq fallback: splitting batch into %d sub-batches", len(groq_sub_batches))
                
                groq_success_count = 0
                for j, g_batch in enumerate(groq_sub_batches):
                    try:
                        results = await _call_groq(g_batch, groq_api_key, groq_model)
                        logger.info("Groq sub-batch %d succeeded: %d claims", j + 1, len(results))
                        if results:
                            raw_claims.extend(results)
                            groq_success_count += len(results)
                    except ClaimExtractionError as g_exc:
                        logger.error("Groq fallback claim extraction failed for sub-batch %d: %s", j + 1, g_exc)
                
                logger.info("Groq fallback claim extraction succeeded: %d claims extracted", groq_success_count)
            else:
                logger.warning("GROQ_API_KEY not configured, skipping fallback.")'''

if old_loop in content:
    content = content.replace(old_loop, new_loop)
else:
    print("Could not find old loop block to replace.")

with open('app/services/claims.py', 'w', encoding='utf-8') as f:
    f.write(content)
