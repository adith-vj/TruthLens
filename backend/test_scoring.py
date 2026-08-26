from app.services.claims import _score_claim

claims = [
    {"text": "The ship was big", "claim_type": "factual_claim", "has_entities": True},
    {"text": "The Titanic sank on April 15 1912", "claim_type": "factual_claim", "has_numbers": True, "has_entities": True, "is_specific": True},
    {"text": "1517 people died when the Titanic sank in the North Atlantic", "claim_type": "factual_claim", "has_numbers": True, "has_entities": True, "is_specific": True},
    {"text": "The ship was probably around 882 feet long", "claim_type": "factual_claim", "has_numbers": True, "has_entities": True, "is_specific": True},
    {"text": "I think the Titanic story is very sad", "claim_type": "factual_claim", "has_entities": True},
]

print(f"{'checkability':>12} | {'claim_score':>11} | {'type':>14} | text")
print("-" * 90)
for c in claims:
    check, score, ct = _score_claim(c)
    print(f"{check:12.3f} | {score:11.3f} | {ct:>14} | {c['text'][:60]}")
