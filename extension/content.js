
function showTruthLensToast(message) {
    let toast = document.getElementById('truthlens-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'truthlens-toast';
        document.body.appendChild(toast);
    }
    toast.innerHTML = `<div class="truthlens-spinner"></div> <span>${message}</span>`;
    // Tiny delay ensures the CSS transition triggers
    setTimeout(() => toast.classList.add('tl-visible'), 10); 
}

function hideTruthLensToast() {
    const toast = document.getElementById('truthlens-toast');
    if (toast) {
        toast.classList.remove('tl-visible');
    }
}

// Track the most recent text selection to ensure it's not lost when the popup opens
let lastSelectedText = "";
document.addEventListener("selectionchange", () => {
    const selection = window.getSelection().toString().trim();
    if (selection) {
        lastSelectedText = selection;
    }
});

// Listens for a message from your popup.js or background.js
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    
    // --- TEXT EXPLAINER LOGIC ---
    if (request.action === "getSelectedText") {
        // Fallback to current selection if lastSelectedText is somehow empty
        const currentSelection = window.getSelection().toString().trim();
        sendResponse({ text: currentSelection || lastSelectedText });
        return true;
    }
    
    if (request.action === "triggerExplainer") {
        console.log("TruthLens: Explainer Mode Activated");
        
        showTruthLensToast("Analyzing article for loaded language...");
        document.body.style.cursor = "progress"; // Change mouse to loading state
        
        const paragraphs = document.querySelectorAll('p'); 
        const scanPromises = []; // Track all our backend requests
        
        paragraphs.forEach(p => {
            if (p.innerText.length > 50) {
                // Push the promise into our tracking array
                scanPromises.push(analyzeAndHighlight(p)); 
            }
        });
        
        // When every single paragraph is done scanning, hide the loader
        Promise.all(scanPromises).then(() => {
            hideTruthLensToast();
            document.body.style.cursor = "default"; // Reset mouse
        });
        
        sendResponse({status: "scanning"});
    }

    // --- NEW: INTERMEDIATE LOADING STATE FOR IMAGES ---
    else if (request.action === "showLoader") {
        showTruthLensToast(request.message);
    }

    // --- IMAGE MODAL UI INJECTION ---
    else if (request.action === "showImageAnalysis") {
        console.log("TruthLens: Injecting Image Forensics UI");
        
        // The backend finished! Hide the toast loader immediately.
        hideTruthLensToast();
        
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

    // --- TEXT CLAIM VERIFICATION UI INJECTION ---
    else if (request.action === "showClaimAnalysis") {
        console.log("TruthLens: Injecting Claim Verification UI");
        
        hideTruthLensToast();
        
        const existingModal = document.getElementById('truthlens-claim-modal');
        if (existingModal) existingModal.remove();

        const overlay = document.createElement('div');
        overlay.id = 'truthlens-claim-modal';
        overlay.className = 'truthlens-overlay';

        const data = request.analysis;
        const claimText = request.claimText;
        
        // Colors mapping similar to the popup styles
        let verdictColor, verdictBg, verdictBorder, verdictText;
        if (data.verdict === 'true') { verdictColor = '#2e7d32'; verdictBg = '#e8f5e9'; verdictBorder = '#c8e6c9'; verdictText = 'True'; }
        else if (data.verdict === 'false') { verdictColor = '#c62828'; verdictBg = '#ffebee'; verdictBorder = '#ffcdd2'; verdictText = 'False'; }
        else if (data.verdict === 'misleading') { verdictColor = '#f57f17'; verdictBg = '#fff8e1'; verdictBorder = '#ffecb3'; verdictText = 'Misleading'; }
        else { verdictColor = '#48484a'; verdictBg = '#f2f2f7'; verdictBorder = '#e5e5ea'; verdictText = 'Unverifiable'; }

        const confidenceStr = data.verdict === 'unverifiable' ? "Opinion or insufficient evidence" : Math.round(data.confidence_score * 100) + "% confidence";
        const displayClaim = claimText.length > 250 ? claimText.substring(0, 250) + "..." : claimText;
        
        let sourcesHtml = '';
        if (data.sources && data.sources.length > 0) {
            sourcesHtml = `
                <div style="font-size: 11px; text-transform: uppercase; font-weight: 600; color: #9ca3af; margin-top: 16px; margin-bottom: 6px;">Sources</div>
                <div style="display: flex; flex-direction: column; gap: 8px; max-height: 200px; overflow-y: auto;">
                    ${data.sources.map(s => `
                        <a href="${s.url}" target="_blank" rel="noopener noreferrer" style="display: block; padding: 10px; background: #fbfbfb; border: 1px solid #e5e5ea; border-radius: 6px; text-decoration: none;">
                            <div style="font-size: 10px; font-weight: 600; color: #636366; margin-bottom: 4px; text-transform: uppercase;">
                                🔗 ${s.publisher || new URL(s.url).hostname}
                            </div>
                            <div style="font-size: 13px; color: #1c1c1e; font-weight: 500; line-height: 1.4;">${s.title}</div>
                        </a>
                    `).join('')}
                </div>
            `;
        } else {
            sourcesHtml = `
                <div style="margin-top: 16px; font-size: 13px; color: #636366; background: #fbfbfb; padding: 12px; border-radius: 6px; border: 1px dashed #e5e5ea; text-align: center;">
                    Available evidence was insufficient to verify this claim or it may be an opinion/advertisement.
                </div>
            `;
        }

        overlay.innerHTML = `
            <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center; z-index: 999999; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
                <div style="background: white; border-radius: 12px; width: 380px; max-width: 90%; padding: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); position: relative;">
                    <button id="tl-claim-close-btn" style="position: absolute; top: 12px; right: 12px; background: transparent; border: none; font-size: 20px; cursor: pointer; color: #aeaeb2;">&times;</button>
                    
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                        <span style="background: ${verdictBg}; color: ${verdictColor}; border: 1px solid ${verdictBorder}; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; text-transform: uppercase;">
                            ${verdictText}
                        </span>
                        <span style="font-size: 12px; color: #636366; font-weight: 500;">
                            ${confidenceStr}
                        </span>
                    </div>

                    <div style="background: #fbfbfb; border: 1px dashed #e5e5ea; border-radius: 6px; padding: 12px; font-size: 14px; color: #636366; line-height: 1.5; margin-bottom: 16px;">
                        "${displayClaim}"
                    </div>
                    
                    ${sourcesHtml}
                </div>
            </div>
        `;
        document.body.appendChild(overlay);

        document.getElementById('tl-claim-close-btn').addEventListener('click', () => {
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