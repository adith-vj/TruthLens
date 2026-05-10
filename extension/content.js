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

    else if (request.action === "showImageAnalysis") {
        console.log("TruthLens: Injecting Image Forensics UI");
        
        // Remove existing modal if one is already open
        const existingModal = document.getElementById('truthlens-image-modal');
        if (existingModal) existingModal.remove();

        // Build the HTML overlay
        const overlay = document.createElement('div');
        overlay.id = 'truthlens-image-modal';
        overlay.className = 'truthlens-overlay';

        const { tier_1_metadata, tier_2_ela } = request.analysis;
        
        overlay.innerHTML = `
            <div class="truthlens-modal">
                <button class="truthlens-close-btn" id="tl-close-btn">&times;</button>
                <div class="truthlens-modal-header" style="display: flex; justify-content: space-between; align-items: center;">
                    <span>📸 TruthLens Image Forensics</span>
                    
                    <button id="tl-deep-scan-btn" style="background: #2563eb; color: white; border: none; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 13px; transition: 0.2s;">
                        🤖 Run Deep AI Scan
                    </button>
                </div>
                
                <div id="tl-deep-scan-results" style="display: none; padding: 10px; background: #0f172a; border-radius: 6px; margin-bottom: 15px; border-left: 4px solid #3b82f6; font-size: 14px;">
                    Scanning pixels...
                </div>

                <div style="margin-bottom: 15px; font-size: 14px; color: #cbd5e1;">
                    <strong>Format:</strong> ${tier_1_metadata.format} | 
                    <strong>Size:</strong> ${tier_1_metadata.size} | 
                    <strong>Metadata Intact:</strong> ${tier_1_metadata.exif_present ? 'Yes' : 'No (Stripped by host)'}
                </div>

                <div class="truthlens-image-container">
                    <div class="truthlens-image-box">
                        <span class="truthlens-tag">Original Image</span>
                        <img src="${request.originalUrl}" alt="Original">
                    </div>
                    
                    <div class="truthlens-image-box">
                        <span class="truthlens-tag">Error Level Analysis (ELA)</span>
                        <img src="${tier_2_ela.heatmap_base64}" alt="ELA Heatmap">
                        <p style="font-size: 12px; color: #94a3b8; text-align: center; margin-top: 10px;">
                            Bright areas indicate varying compression levels, which may suggest digital splicing or manipulation.
                        </p>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(overlay);

        // Wire up the close button
        document.getElementById('tl-close-btn').addEventListener('click', () => {
            overlay.remove();
        });
        document.getElementById('tl-deep-scan-btn').addEventListener('click', async () => {
            const btn = document.getElementById('tl-deep-scan-btn');
            const resultsDiv = document.getElementById('tl-deep-scan-results');
            
            btn.style.display = 'none'; // Hide the button
            resultsDiv.style.display = 'block'; // Show the scanning text
            
            try {
                const response = await fetch("http://127.0.0.1:8000/deep-scan-image", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image_url: request.originalUrl })
                });

                if (!response.ok) throw new Error("Backend connection failed");
                const data = await response.json();
                
                // Format the results
                const isFake = data.prediction === 'FAKE';
                const color = isFake ? '#ef4444' : '#22c55e'; // Red if fake, Green if real
                
                resultsDiv.style.borderLeftColor = color;
                resultsDiv.innerHTML = `<strong>AI Conclusion:</strong> <span style="color: ${color};">${data.prediction}</span> (Confidence: ${data.confidence}%)`;

            } catch (error) {
                resultsDiv.innerHTML = `<span style="color: #ef4444;">Error: Could not complete deep scan.</span>`;
            }
        });
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
        const flaggedSentences = data.flagged_sentences; 
        
        let highlightedHtml = originalText;
        
        flaggedSentences.forEach(sentence => {
            // 1. Escape the special characters
            let escapedSentence = sentence.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            
            // 2. THE FIX: Replace hard spaces with a flexible regex space matcher (\s+)
            // This tells JS: "Match this sentence even if the browser added weird hidden line-breaks or extra spaces"
            escapedSentence = escapedSentence.replace(/\s+/g, '\\s+');
            
            const regex = new RegExp(escapedSentence, 'g'); 
            
            highlightedHtml = highlightedHtml.replace(regex, (match) => {
                return `<span class="truthlens-highlight" 
                    style="background-color: rgba(255, 69, 58, 0.15); border-bottom: 2px dotted rgba(255, 69, 58, 0.6); cursor: help; border-radius: 3px;" 
                    title="TruthLens Flag: High concentration of emotionally loaded or subjective language.">
                    ${match}
                </span>`;
            });
        });
        targetElement.innerHTML = highlightedHtml;

    } catch (error) {
        console.error("TruthLens Error:", error);
    }
}