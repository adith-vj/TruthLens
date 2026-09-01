import re

with open('app/tests/test_claims.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_tests = '''
@pytest.mark.asyncio
async def test_groq_batch_splitting():
    # Test that a large batch gets split into smaller sub-batches
    from app.services.claims import _chunk_text_for_groq
    
    # Create lines of 100 characters each
    line = "x" * 95 + " [0-1]"
    text = "\\n".join([line for _ in range(10)]) # Total approx 1000 chars
    
    # Chunk with max_chars = 300
    batches = _chunk_text_for_groq(text, 300)
    
    # 10 lines of 102 chars = 1020 chars. Each batch can hold 2 lines (204 chars).
    # So we expect 5 batches.
    assert len(batches) == 5
    for b in batches:
        assert len(b) <= 300
        assert "[0-1]" in b

@pytest.mark.asyncio
async def test_process_video_claims_groq_subbatch_partial_success():
    from app.services.claims import ClaimExtractionError
    
    mock_transcript = {"segments": [_seg("anything", 0, 1)]}
    
    # We will mock _chunk_text_for_groq to return 2 sub-batches
    with patch("app.services.claims.get_transcript", return_value=mock_transcript), \\
         patch("app.services.claims.settings") as mock_settings, \\
         patch("app.services.claims._chunk_text_for_groq", return_value=["batch1", "batch2"]), \\
         patch("app.services.claims._call_gemini", side_effect=ClaimExtractionError("429")), \\
         patch("app.services.claims._call_groq") as mock_groq:
        
        mock_settings.GEMINI_API_KEY.get_secret_value.return_value = "test-key"
        mock_settings.GROQ_API_KEY.get_secret_value.return_value = "groq-key"
        mock_settings.GROQ_MODEL = "llama-3.3-70b-versatile"
        
        # Sub-batch 1 succeeds, Sub-batch 2 fails
        mock_groq.side_effect = [
            [_make_raw("groq claim 1", start=1.0, end=2.0)],
            ClaimExtractionError("Sub-batch 2 failed")
        ]
        
        claims = await process_video_claims("FAKE_VIDEO_ID")
        
    assert len(claims) == 1
    assert claims[0].text == "groq claim 1"
    assert mock_groq.call_count == 2

@pytest.mark.asyncio
async def test_groq_empty_content_controlled_failure():
    from app.services.claims import _call_groq, ClaimExtractionError
    
    # Mock httpx response to return 200 OK but empty content
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": ""}}]
    }
    
    with patch("httpx.AsyncClient.post", return_value=mock_response):
        with pytest.raises(ClaimExtractionError, match="Groq returned empty content"):
            await _call_groq("some transcript", "fake-key", "fake-model")

@pytest.mark.asyncio
async def test_groq_http_413_controlled_failure():
    from app.services.claims import _call_groq, ClaimExtractionError
    import httpx
    
    # Mock httpx response to raise HTTPStatusError
    mock_response = httpx.Response(413, json={"error": "Too large"})
    
    async def mock_post(*args, **kwargs):
        raise httpx.HTTPStatusError("413 Error", request=AsyncMock(), response=mock_response)
        
    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        with pytest.raises(ClaimExtractionError, match="Groq HTTP error 413"):
            await _call_groq("some transcript", "fake-key", "fake-model")

@pytest.mark.asyncio
async def test_groq_malformed_json_controlled_failure():
    from app.services.claims import _call_groq, ClaimExtractionError
    
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "This is not JSON at all"}}]
    }
    
    with patch("httpx.AsyncClient.post", return_value=mock_response):
        with pytest.raises(ClaimExtractionError):
            await _call_groq("some transcript", "fake-key", "fake-model")
'''

content = content.replace('_make_raw', '_raw')

if 'test_groq_batch_splitting' not in content:
    content += "\n" + new_tests

with open('app/tests/test_claims.py', 'w', encoding='utf-8') as f:
    f.write(content)
