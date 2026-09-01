import re

with open('app/tests/test_claims.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('_make_raw', '_raw')

old_empty_test = '''    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": ""}}]
    }'''

new_empty_test = '''    from unittest.mock import MagicMock
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": ""}}]
    }'''
content = content.replace(old_empty_test, new_empty_test)

old_malformed_test = '''    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "This is not JSON at all"}}]
    }'''

new_malformed_test = '''    from unittest.mock import MagicMock
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "This is not JSON at all"}}]
    }'''
content = content.replace(old_malformed_test, new_malformed_test)

with open('app/tests/test_claims.py', 'w', encoding='utf-8') as f:
    f.write(content)
