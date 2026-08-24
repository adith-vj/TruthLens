// 1. Create the context menus when the extension is installed or updated
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "verify-image-truthlens",
    title: "🔍 Verify Image with TruthLens",
    contexts: ["image"]
  });
  
  chrome.contextMenus.create({
    id: "verify-text-truthlens",
    title: "🔍 Verify Claim with TruthLens",
    contexts: ["selection"]
  });
});

const activeVerifications = new Set();

// 2. Listen for when the user clicks our menu items
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  
  // IMAGE FORENSICS FLOW
  if (info.menuItemId === "verify-image-truthlens") {
    const imageUrl = info.srcUrl;
    console.log("TruthLens: Target acquired ->", imageUrl);

    chrome.tabs.sendMessage(tab.id, { 
        action: "showLoader", 
        message: "Extracting image forensics..." 
    });

    try {
      const response = await fetch("http://127.0.0.1:8000/analyze-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_url: imageUrl })
      });

      if (!response.ok) throw new Error("Failed to connect to FastAPI backend");
      const data = await response.json();

      chrome.tabs.sendMessage(tab.id, {
        action: "showImageAnalysis",
        originalUrl: imageUrl,
        analysis: data
      });

    } catch (error) {
      console.error("TruthLens Error:", error.message);
      chrome.tabs.sendMessage(tab.id, { 
        action: "showLoader", 
        message: "Error: Connection to TruthLens failed." 
      });
      setTimeout(() => chrome.tabs.sendMessage(tab.id, { action: "hideLoader" }), 3000);
    }
  }

  // TEXT CLAIM VERIFICATION FLOW
  if (info.menuItemId === "verify-text-truthlens") {
    if (activeVerifications.has(tab.id)) {
        console.log("TruthLens: Verification already in progress for this tab.");
        return;
    }
    
    // Phase 5.1: Request accurate selection from content.js
    chrome.tabs.sendMessage(tab.id, { action: "getSelectedText" }, async (response) => {
        let claimText = response?.text || info.selectionText;
        
        if (!claimText || !claimText.trim()) {
            chrome.tabs.sendMessage(tab.id, { 
                action: "showLoader", 
                message: "Error: No text selected for verification." 
            });
            setTimeout(() => chrome.tabs.sendMessage(tab.id, { action: "hideLoader" }), 3000);
            return;
        }

        // Clean up transcript text: remove timestamps like "0:00" or "1:23:45"
        claimText = claimText.replace(/\b\d{1,2}:\d{2}(:\d{2})?\b/g, '');
        // Replace newlines and multiple spaces with a single space
        claimText = claimText.replace(/[\r\n]+/g, ' ').replace(/\s+/g, ' ').trim();
        
        if (claimText.length > 2000) {
            claimText = claimText.substring(0, 2000) + "...";
        }
        
        if (!claimText) {
            chrome.tabs.sendMessage(tab.id, { 
                action: "showLoader", 
                message: "Error: Selected text did not contain a valid claim." 
            });
            setTimeout(() => chrome.tabs.sendMessage(tab.id, { action: "hideLoader" }), 3000);
            return;
        }

        console.log("TruthLens: Verifying Claim ->", claimText);
        activeVerifications.add(tab.id);

        chrome.tabs.sendMessage(tab.id, { 
            action: "showLoader", 
            message: "Analyzing claim and checking sources..." 
        });

        try {
          const apiResponse = await fetch("http://127.0.0.1:8000/api/verify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: claimText })
          });

          if (!apiResponse.ok) throw new Error("Failed to connect to verification server");
          const data = await apiResponse.json();

          chrome.tabs.sendMessage(tab.id, {
            action: "showClaimAnalysis",
            claimText: claimText,
            analysis: data
          });

        } catch (error) {
          console.error("TruthLens Error:", error.message);
          chrome.tabs.sendMessage(tab.id, { 
            action: "showLoader", 
            message: "Error: Connection to TruthLens failed." 
          });
          setTimeout(() => chrome.tabs.sendMessage(tab.id, { action: "hideLoader" }), 3000);
        } finally {
          activeVerifications.delete(tab.id);
        }
    });
  }
});



