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
                <div class="truthlens-modal-header">📸 TruthLens Image Forensics</div>
                
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
            if (token.weight > 0.5) { 
                const alpha = token.weight * 0.5; 
                const tooltipText = `TruthLens Flag: Loaded language (Confidence: ${Math.round(token.weight * 100)}%)`;
                
                // CRITICAL FIX: Removed the extra space at the end of the string
                highlightedHtml += `<span class="truthlens-highlight" style="background-color: rgba(255, 69, 58, ${alpha});" title="${tooltipText}">${token.word}</span>`;
            } else {
                // CRITICAL FIX: Removed the extra space here too
                highlightedHtml += token.word; 
            }
        });

        // CRITICAL FIX: Removed .trim() so we don't accidentally cut off leading/trailing formatting
        targetElement.innerHTML = highlightedHtml;

    } catch (error) {
        console.error("TruthLens Error:", error);
    }
}