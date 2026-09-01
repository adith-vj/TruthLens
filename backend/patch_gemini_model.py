import re

with open('app/services/claims.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_url = '"/v1beta/models/gemini-3.6-flash:generateContent"'
new_url = '"/v1beta/models/gemini-3.5-flash-lite:generateContent"'

content = content.replace(old_url, new_url)

with open('app/services/claims.py', 'w', encoding='utf-8') as f:
    f.write(content)
