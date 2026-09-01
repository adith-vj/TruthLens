import re
import os

files_to_patch = ['app/services/classifier.py', 'app/services/llm.py']
old_url = '"/v1beta/models/gemini-3.6-flash:generateContent"'
new_url = '"/v1beta/models/gemini-3.5-flash-lite:generateContent"'

for file_path in files_to_patch:
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old_url in content:
            content = content.replace(old_url, new_url)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
