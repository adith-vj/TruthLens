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
    
    // info.srcUrl contains the direct link to the image
    const imageUrl = info.srcUrl;
    console.log("TruthLens: Target acquired ->", imageUrl);

    try {
      // 3. Fire the URL off to our (soon-to-be-built) FastAPI endpoint
      const response = await fetch("http://127.0.0.1:8000/analyze-image", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ image_url: imageUrl })
      });

      if (!response.ok) throw new Error("Failed to connect to FastAPI backend");

      const data = await response.json();
      chrome.tabs.sendMessage(tab.id, {
        action: "showImageAnalysis",
        originalUrl: imageUrl,
        analysis: data
      });
      console.log("TruthLens Image Forensics:", data);
      console.log("TruthLens Image Forensics:\n", JSON.stringify(data, null, 2));

      // Note: We are just logging to the console for now. 
      // Later, we can inject a popup or a tooltip directly onto the image!

    } catch (error) {
      console.error("TruthLens Error:", error.message);
    }
  }
});