import re

with open('app/tests/test_claims.py', 'r', encoding='utf-8') as f:
    content = f.read()

groq_tests = '''
@pytest.mark.asyncio
async def test_process_video_claims_gemini_fails_groq_succeeds():
    mock_transcript = {"segments": [_seg("anything", 0, 1)]}
    groq_response = [_make_raw("groq claim", start=1.0, end=2.0)]
    
    from app.services.claims import ClaimExtractionError

    with patch("app.services.claims.get_transcript", return_value=mock_transcript), \\
         patch("app.services.claims.settings") as mock_settings, \\
         patch("app.services.claims._call_gemini", side_effect=ClaimExtractionError("429")), \\
         patch("app.services.claims._call_groq", new=AsyncMock(return_value=groq_response)) as mock_groq:
        
        mock_settings.GEMINI_API_KEY.get_secret_value.return_value = "test-key"
        mock_settings.GROQ_API_KEY.get_secret_value.return_value = "groq-key"
        mock_settings.GROQ_MODEL = "llama-3.3-70b-versatile"
        
        claims = await process_video_claims("FAKE_VIDEO_ID")
        
    assert len(claims) == 1
    assert claims[0].text == "groq claim"
    mock_groq.assert_called_once()


@pytest.mark.asyncio
async def test_process_video_claims_gemini_succeeds_groq_not_called():
    mock_transcript = {"segments": [_seg("anything", 0, 1)]}
    gemini_response = [_make_raw("gemini claim", start=1.0, end=2.0)]
    
    with patch("app.services.claims.get_transcript", return_value=mock_transcript), \\
         patch("app.services.claims.settings") as mock_settings, \\
         patch("app.services.claims._call_gemini", new=AsyncMock(return_value=gemini_response)), \\
         patch("app.services.claims._call_groq") as mock_groq:
        
        mock_settings.GEMINI_API_KEY.get_secret_value.return_value = "test-key"
        mock_settings.GROQ_API_KEY.get_secret_value.return_value = "groq-key"
        mock_settings.GROQ_MODEL = "llama-3.3-70b-versatile"
        
        claims = await process_video_claims("FAKE_VIDEO_ID")
        
    assert len(claims) == 1
    assert claims[0].text == "gemini claim"
    mock_groq.assert_not_called()


@pytest.mark.asyncio
async def test_process_video_claims_both_fail():
    mock_transcript = {"segments": [_seg("anything", 0, 1)]}
    
    from app.services.claims import ClaimExtractionError

    with patch("app.services.claims.get_transcript", return_value=mock_transcript), \\
         patch("app.services.claims.settings") as mock_settings, \\
         patch("app.services.claims._call_gemini", side_effect=ClaimExtractionError("gemini fail")), \\
         patch("app.services.claims._call_groq", side_effect=ClaimExtractionError("groq fail")) as mock_groq:
        
        mock_settings.GEMINI_API_KEY.get_secret_value.return_value = "test-key"
        mock_settings.GROQ_API_KEY.get_secret_value.return_value = "groq-key"
        mock_settings.GROQ_MODEL = "llama-3.3-70b-versatile"
        
        claims = await process_video_claims("FAKE_VIDEO_ID")
        
    assert claims == []
    mock_groq.assert_called_once()
'''

if "test_process_video_claims_both_fail" not in content:
    content += "\n" + groq_tests

with open('app/tests/test_claims.py', 'w', encoding='utf-8') as f:
    f.write(content)
