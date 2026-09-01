import re

# 1. Update classifier.py
with open('app/services/classifier.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add ClassifierQuotaError
if 'class ClassifierQuotaError' not in content:
    content = content.replace(
        'class ClassifierConfigError',
        'class ClassifierQuotaError(ClassifierError):\n    """Raised when Gemini returns HTTP 429."""\n\n\nclass ClassifierConfigError'
    )
    content = content.replace('"ClassifierConfigError",', '"ClassifierConfigError",\n    "ClassifierQuotaError",')

# Raise it in _classify_with_gemini
if 'if response.status_code == 429:' not in content:
    content = content.replace(
        'response.raise_for_status()',
        'if response.status_code == 429:\n                raise ClassifierQuotaError("Gemini classification quota exceeded (HTTP 429)")\n            response.raise_for_status()'
    )

# Do NOT swallow it in classify_claim
if 'except ClassifierQuotaError:' not in content:
    content = content.replace(
        'except ClassifierConfigError:',
        'except ClassifierQuotaError:\n        raise\n    except ClassifierConfigError:'
    )

with open('app/services/classifier.py', 'w', encoding='utf-8') as f:
    f.write(content)

# 2. Update app/api/verify.py
with open('app/api/verify.py', 'r', encoding='utf-8') as f:
    content = f.read()

if 'ClassifierQuotaError' not in content:
    content = content.replace(
        'from app.services.classifier import ClaimType, classify_claim',
        'from app.services.classifier import ClaimType, classify_claim, ClassifierQuotaError'
    )
    content = content.replace(
        'claim_type = await classify_claim(claim_text)',
        'try:\n        claim_type = await classify_claim(claim_text)\n    except ClassifierQuotaError:\n        logger.warning("Gemini classifier quota exceeded - returning unverifiable")\n        return _unverifiable()'
    )

with open('app/api/verify.py', 'w', encoding='utf-8') as f:
    f.write(content)

# 3. Update app/services/video_verify.py
with open('app/services/video_verify.py', 'r', encoding='utf-8') as f:
    content = f.read()

if 'raise LLMQuotaError("Gemini first-pass quota exceeded' not in content:
    content = content.replace(
        'if response.status_code == 429:\n        logger.warning("Gemini first-pass quota exceeded")\n        return FirstPassResult(verdict="uncertain", confidence=0.0, needs_web_search=True)',
        'if response.status_code == 429:\n        logger.warning("Gemini first-pass quota exceeded")\n        raise LLMQuotaError("Gemini first-pass quota exceeded (HTTP 429)")'
    )

if 'except LLMQuotaError' not in content.split('first_pass = await _gemini_first_pass')[1][:200]:
    content = content.replace(
        'first_pass = await _gemini_first_pass(claim_text, context)',
        'try:\n        first_pass = await _gemini_first_pass(claim_text, context)\n    except LLMQuotaError:\n        logger.warning("LLM first-pass quota exceeded - unverifiable")\n        return _cache_and_return(VideoVerifyResult(\n            verdict="unverifiable", confidence_score=0.0, sources=[], metrics=metrics,\n        ))'
    )

with open('app/services/video_verify.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patched.")
