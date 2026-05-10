// 1. Create the context menu when the extension is installed or updated
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "verify-image-truthlens",
    title: "🔍 Verify Image with TruthLens",
    contexts: ["image"] // This ensures it ONLY appears when right-clicking an image
  });
});

// 2. Listen for when the user clicks our new menu item
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "verify-image-truthlens") {
    const imageUrl = info.srcUrl;
    console.log("TruthLens: Target acquired ->", imageUrl);

    // NEW: Instantly tell the content script to pop the loading toast
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
      // Optional: Tell the user if the server is offline
      chrome.tabs.sendMessage(tab.id, { 
        action: "showLoader", 
        message: "Error: Connection to TruthLens failed." 
      });
      setTimeout(() => chrome.tabs.sendMessage(tab.id, { action: "hideLoader" }), 3000);
    }
  }
});