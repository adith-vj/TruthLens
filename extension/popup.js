document.getElementById('analyzeBtn').addEventListener('click', async () => {
  const statusText = document.getElementById('statusText');
  const resultsDiv = document.getElementById('results');
  
  statusText.innerText = 'Scanning page...';
  resultsDiv.style.display = 'none';

  // 1. Get the current active tab (Text AND URL)
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    function: scrapePageText,
  }, async (injectionResults) => {
    
    if (!injectionResults || !injectionResults[0].result) {
        statusText.innerText = 'Could not read page text.';
        return;
    }

    const scrapedText = injectionResults[0].result;

    try {
      statusText.innerText = 'Analyzing via TruthLens AI...';
      
      // 2. Send both the TEXT and the URL to the FastAPI backend
      const response = await fetch('http://127.0.0.1:8000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          text: scrapedText, 
          url: tab.url // Grabbed directly from the Chrome tab!
        })
      });
      
      const data = await response.json();
      
      // 3. Populate the UI elements
      document.getElementById('sourceVal').innerText = `${data.source} (${data.historic_bias})`;
      
      // Fake News Logic (Remember: LABEL_1 is Real, LABEL_0 is Fake based on our earlier test)
      const isReal = data.ai_analysis.fake_news.label === 'LABEL_1';
      const fakeConf = Math.round(data.ai_analysis.fake_news.score * 100);
      document.getElementById('fakeVal').innerText = isReal ? `✅ Credible (${fakeConf}%)` : `🚨 Fake/Unreliable (${fakeConf}%)`;

      // Tone Logic
      const toneLabel = data.ai_analysis.tone_bias.label; // Will be "NEUTRAL" or "BIASED"
      const toneConf = Math.round(data.ai_analysis.tone_bias.score * 100);
      document.getElementById('biasVal').innerText = toneLabel === 'NEUTRAL' ? `⚖️ Objective (${toneConf}%)` : `🎭 Highly Subjective (${toneConf}%)`;

      // Hide the loading text and show the results dashboard
      statusText.innerText = '';
      resultsDiv.style.display = 'block';
      
    } catch (err) {
      statusText.innerText = 'Error: Make sure FastAPI is running.';
      console.error(err);
    }
  });
});

function scrapePageText() {
  let paragraphs = document.getElementsByTagName('p');
  let text = '';
  for (let p of paragraphs) {
    text += p.innerText + ' ';
  }
  return text.substring(0, 2000); 
}
// --- NEW EXPLAINER MODE LOGIC ---
document.getElementById('explainBtn').addEventListener('click', async () => {
  const statusText = document.getElementById('statusText');
  statusText.innerText = 'Highlighting loaded words on page...';

  // Get the active tab
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  // Send a message to content.js injected in that tab
  chrome.tabs.sendMessage(tab.id, { action: "triggerExplainer" }, (response) => {
    // Catch errors (like if the user is on a protected chrome:// page)
    if (chrome.runtime.lastError) {
      statusText.innerText = 'Error: Try reloading the page.';
      console.error(chrome.runtime.lastError.message);
    } else if (response && response.status === "scanning") {
      statusText.innerText = '✨ Look at the webpage! Highlights applied.';
    }
  });
});