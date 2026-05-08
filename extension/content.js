// Listens for a message from your popup.js to trigger the scan
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "triggerExplainer") {
        console.log("TruthLens: Explainer Mode Activated");
        
        // Let's grab the main paragraphs of the article to scan
        // This targets standard article text without grabbing menus/footers
        const paragraphs = document.querySelectorAll('p'); 
        
        paragraphs.forEach(p => {
            // Only process paragraphs with actual substance
            if (p.innerText.length > 50) {
                analyzeAndHighlight(p);
            }
        });
        
        sendResponse({status: "scanning"});
    }
    return true; 
});

async function analyzeAndHighlight(targetElement) {
    const originalText = targetElement.innerText;

    try {
        const response = await fetch("http://127.0.0.1:8000/explain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: originalText })
        });

        if (!response.ok) throw new Error("Backend connection failed");
        
        const data = await response.json();
        let highlightedHtml = "";
        
        data.tokens.forEach(token => {
            if (token.weight > 0.5) { // Threshold for loaded language
                const alpha = token.weight * 0.5; // Scale down opacity so it's readable
                const tooltipText = `TruthLens Flag: Loaded language (Confidence: ${Math.round(token.weight * 100)}%)`;
                
                highlightedHtml += `<span class="truthlens-highlight" style="background-color: rgba(255, 69, 58, ${alpha});" title="${tooltipText}">${token.word}</span> `;
            } else {
                highlightedHtml += `${token.word} `;
            }
        });

        targetElement.innerHTML = highlightedHtml.trim();

    } catch (error) {
        console.error("TruthLens Error:", error);
    }
}