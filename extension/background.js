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
    const claimText = info.selectionText;
    console.log("TruthLens: Verifying Claim ->", claimText);

    chrome.tabs.sendMessage(tab.id, { 
        action: "showLoader", 
        message: "Analyzing claim and checking sources..." 
    });

    try {
      const response = await fetch("http://127.0.0.1:8000/api/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: claimText })
      });

      if (!response.ok) throw new Error("Failed to connect to verification server");
      const data = await response.json();

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
    }
  }
});