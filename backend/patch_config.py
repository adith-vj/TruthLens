import re

with open('app/core/config.py', 'r', encoding='utf-8') as f:
    content = f.read()

if 'GROQ_MAX_INPUT_TOKENS' not in content:
    old_groq = '    GROQ_MODEL: str = "llama-3.3-70b-versatile"'
    new_groq = '    GROQ_MODEL: str = "llama-3.3-70b-versatile"\n    # Target max input tokens for Groq to avoid 413s on restricted tiers (e.g. openai/gpt-oss-120b limit is 8000)\n    GROQ_MAX_INPUT_TOKENS: int = 5000'
    content = content.replace(old_groq, new_groq)
    with open('app/core/config.py', 'w', encoding='utf-8') as f:
        f.write(content)
