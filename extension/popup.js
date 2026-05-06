document.getElementById('analyzeBtn').addEventListener('click', async () => {
  const resultText = document.getElementById('result');
  resultText.innerText = 'Analyzing page...';

  // 1. Get the current active tab
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  // 2. Inject a script into that tab to read the webpage text
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    function: scrapePageText,
  }, async (injectionResults) => {
    
    // Safety check in case it couldn't read the page
    if (!injectionResults || !injectionResults[0].result) {
        resultText.innerText = 'Could not read page text.';
        return;
    }

    const scrapedText = injectionResults[0].result;

    // 3. Send the scraped text to your FastAPI backend
    try {
      const response = await fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: scrapedText })
      });
      
      const data = await response.json();
      
      // The model uses LABEL_0 for Fake and LABEL_1 for Real
      const isFake = data.result.label === 'LABEL_0';
      const labelText = isFake ? '🚨 High Bias / Fake News' : '✅ Looks Credible';
      const confidence = Math.round(data.result.score * 100);
      
      resultText.innerText = `${labelText} \n(${confidence}% confidence)`;
      
    } catch (err) {
      resultText.innerText = 'Error: Make sure FastAPI is running.';
      console.error(err);
    }
  });
});

// This function runs directly inside the webpage the user is looking at
function scrapePageText() {
  // Grab all paragraph elements on the news site
  let paragraphs = document.getElementsByTagName('p');
  let text = '';
  for (let p of paragraphs) {
    text += p.innerText + ' ';
  }
  // Return just the first 2000 characters to keep it under BERT's token limit
  return text.substring(0, 2000); 
}