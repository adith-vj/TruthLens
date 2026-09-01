import re

with open('app/services/claims.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('logger.info("Groq model configured:', 'logger.error("Groq model configured:')
content = content.replace('logger.info("Groq HTTP status:', 'logger.error("Groq HTTP status:')
content = content.replace("logger.info(\"Groq response contains 'choices':", "logger.error(\"Groq response contains 'choices':")
content = content.replace('logger.info("Groq number of choices:', 'logger.error("Groq number of choices:')
content = content.replace("logger.info(\"Groq choice 0 contains 'message':", "logger.error(\"Groq choice 0 contains 'message':")
content = content.replace('logger.info("Groq message.content type:', 'logger.error("Groq message.content type:')
content = content.replace('logger.info("Groq message.content length:', 'logger.error("Groq message.content length:')
content = content.replace('logger.info("Groq message.content preview:', 'logger.error("Groq message.content preview:')

with open('app/services/claims.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated claims.py to use logger.error for diagnostics")
