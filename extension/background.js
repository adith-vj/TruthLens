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

  chrome.contextMenus.create({
    id: "verify-video-truthlens",
    title: "🔍 Scan Video for Deepfakes (TruthLens)",
    contexts: ["video", "link"]
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

  // VIDEO FORENSICS FLOW (NO AUDIO / VISUAL DEEPFAKE SCAN)
  if (info.menuItemId === "verify-video-truthlens") {
    const videoSrc = info.srcUrl || info.linkUrl;
    console.log("TruthLens: Target video acquired ->", videoSrc);

    // If clicked on a video link (e.g. YouTube thumbnail), directly send to /api/video/analyze-visual-url
    if (info.linkUrl && (info.linkUrl.includes("youtube.com") || info.linkUrl.includes("youtu.be"))) {
        chrome.tabs.sendMessage(tab.id, { 
            action: "showLoader", 
            message: "Fetching YouTube video frames via yt-dlp..." 
        });

        fetch("http://127.0.0.1:8000/api/video/analyze-visual-url", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ video_url: info.linkUrl, max_frames: 12 })
        }).then(async res => {
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || "Server error");
            }
            return res.json();
        }).then(data => {
            chrome.tabs.sendMessage(tab.id, { action: "hideLoader" });
            chrome.tabs.sendMessage(tab.id, {
                action: "showVideoAnalysis",
                analysis: data,
                videoTitle: tab.title || "YouTube Video"
            });
        }).catch(err => {
            console.error("Link video scan error:", err);
            chrome.tabs.sendMessage(tab.id, {
                action: "showLoader",
                message: `Error: ${err.message || "Failed to analyze video."}`,
                autoDismissMs: 3500
            });
            setTimeout(() => {
                chrome.tabs.sendMessage(tab.id, { action: "hideLoader" }, () => {
                    if (chrome.runtime.lastError) {}
                });
            }, 3500);
        });
        return;
    }

    chrome.tabs.sendMessage(tab.id, { 
        action: "showLoader", 
        message: "Extracting video frames for deepfake scan..." 
    });

    chrome.tabs.sendMessage(tab.id, { action: "captureVideoFrames", maxFrames: 12 }, async (resp) => {
        if (resp && resp.status === "success" && resp.frames && resp.frames.length > 0) {
            try {
                const apiRes = await fetch("http://127.0.0.1:8000/api/video/analyze-visual-frames", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        video_id: "context_menu_video",
                        title: tab.title || "Web Video",
                        frames: resp.frames
                    })
                });

                if (!apiRes.ok) throw new Error("Backend visual scan failed");
                const data = await apiRes.json();

                chrome.tabs.sendMessage(tab.id, { action: "hideLoader" });
                chrome.tabs.sendMessage(tab.id, {
                    action: "showVideoAnalysis",
                    analysis: data,
                    videoTitle: tab.title || "Video"
                });
            } catch (err) {
                console.error("Video scan error:", err);
                chrome.tabs.sendMessage(tab.id, {
                    action: "showLoader",
                    message: "Error: Failed to analyze video frames.",
                    autoDismissMs: 3000
                });
                setTimeout(() => {
                    chrome.tabs.sendMessage(tab.id, { action: "hideLoader" }, () => {
                        if (chrome.runtime.lastError) {}
                    });
                }, 3000);
            }
        } else if (videoSrc) {
            try {
                const apiRes = await fetch("http://127.0.0.1:8000/api/video/analyze-visual-url", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ video_url: videoSrc, max_frames: 12 })
                });

                if (!apiRes.ok) throw new Error("Stream scan failed");
                const data = await apiRes.json();

                chrome.tabs.sendMessage(tab.id, { action: "hideLoader" });
                chrome.tabs.sendMessage(tab.id, {
                    action: "showVideoAnalysis",
                    analysis: data,
                    videoTitle: tab.title || "Video"
                });
            } catch (err) {
                chrome.tabs.sendMessage(tab.id, {
                    action: "showLoader",
                    message: "Error: Could not process video stream.",
                    autoDismissMs: 3000
                });
                setTimeout(() => {
                    chrome.tabs.sendMessage(tab.id, { action: "hideLoader" }, () => {
                        if (chrome.runtime.lastError) {}
                    });
                }, 3000);
            }
        } else {
            chrome.tabs.sendMessage(tab.id, {
                action: "showLoader",
                message: "Could not capture video frames.",
                autoDismissMs: 3000
            });
            setTimeout(() => {
                chrome.tabs.sendMessage(tab.id, { action: "hideLoader" }, () => {
                    if (chrome.runtime.lastError) {}
                });
            }, 3000);
        }
    });
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



