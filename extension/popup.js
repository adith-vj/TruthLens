/**
 * @typedef {Object} SourceItem
 * @property {string} title
 * @property {string} url
 * @property {string} publisher
 */

/**
 * @typedef {Object} VerifyResponse
 * @property {"true" | "false" | "misleading" | "unverifiable"} verdict
 * @property {number} confidence_score
 * @property {SourceItem[]} sources
 */

document.addEventListener('DOMContentLoaded', async () => {
    const states = {
        idle: document.getElementById('state-idle'),
        ready: document.getElementById('state-ready'),
        videoReady: document.getElementById('state-video-ready'),
        videoNoTranscript: document.getElementById('state-video-no-transcript'),
        analyzingVideo: document.getElementById('state-analyzing-video'),
        transcriptReady: document.getElementById('state-transcript-ready'),
        videoJob: document.getElementById('state-video-job'),
        visualResult: document.getElementById('state-visual-result'),
        verifying: document.getElementById('state-verifying'),
        result: document.getElementById('state-result'),
        error: document.getElementById('state-error')
    };

    const ui = {
        readyClaimText: document.getElementById('ready-claim-text'),
        verifyBtn: document.getElementById('verify-btn'),
        resultHeader: document.getElementById('result-header'),
        verdictBadge: document.getElementById('verdict-badge'),
        confidenceScore: document.getElementById('confidence-score'),
        resultClaimText: document.getElementById('result-claim-text'),
        sourcesContainer: document.getElementById('sources-container'),
        sourcesList: document.getElementById('sources-list'),
        unverifiableExplainer: document.getElementById('unverifiable-explainer'),
        resetBtn: document.getElementById('reset-btn'),
        retryBtn: document.getElementById('retry-btn'),
        errorMessage: document.getElementById('error-message'),
        videoTranscriptInfo: document.getElementById('video-transcript-info'),
        analyzeVideoBtn: document.getElementById('analyze-video-btn'),
        analyzeVisualReadyBtn: document.getElementById('analyze-visual-ready-btn'),
        analyzeVisualBtn: document.getElementById('analyze-visual-btn'),
        transcriptStatsInfo: document.getElementById('transcript-stats-info'),
        transcriptAutoInfo: document.getElementById('transcript-auto-info'),
        continueVideoBtn: document.getElementById('continue-video-btn'),
        videoJobStats: document.getElementById('video-job-stats'),
        videoJobClaims: document.getElementById('video-job-claims'),
        analyzingVideoTitle: document.getElementById('analyzing-video-title'),
        analyzingVideoSubtitle: document.getElementById('analyzing-video-subtitle'),
        visualVerdictBadge: document.getElementById('visual-verdict-badge'),
        visualConfidenceScore: document.getElementById('visual-confidence-score'),
        visualFlaggedBanner: document.getElementById('visual-flagged-banner'),
        visualFlaggedTimes: document.getElementById('visual-flagged-times'),
        riskFace: document.getElementById('risk-face'),
        riskSeam: document.getElementById('risk-seam'),
        riskScene: document.getElementById('risk-scene'),
        riskFft: document.getElementById('risk-fft'),
        riskTemporal: document.getElementById('risk-temporal'),
        visualFrameCount: document.getElementById('visual-frame-count'),
        visualTimelineList: document.getElementById('visual-timeline-list'),
        visualExplanation: document.getElementById('visual-explanation'),
        visualResetBtn: document.getElementById('visual-reset-btn'),
        localVideoFile: document.getElementById('local-video-file')
    };

    let selectedClaim = "";
    let currentCapability = null;
    let isAnalyzingVideo = false;
    let pollInterval = null;

    // Switch visible state
    function showState(stateName) {
        Object.values(states).forEach(el => { if(el) el.classList.add('hidden'); });
        if(states[stateName]) states[stateName].classList.remove('hidden');
    }

    // 1. Initialize Extension
    try {
        let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        
        // Cannot run script on chrome:// URLs
        if (tab.url.startsWith('chrome://')) {
            showState('idle');
            return;
        }

        // Check for text selection first
        chrome.tabs.sendMessage(tab.id, { action: "getSelectedText" }, (response) => {
            if (chrome.runtime.lastError) {
                console.error("Content script not responding:", chrome.runtime.lastError.message);
                showState('idle');
                return;
            }

            if (response && response.text) {
                selectedClaim = response.text;
                
                // Truncate if insanely long just for UI display
                const displayClaim = selectedClaim.length > 300 
                    ? selectedClaim.substring(0, 300) + "..." 
                    : selectedClaim;
                    
                ui.readyClaimText.textContent = `"${displayClaim}"`;
                showState('ready');
            } else {
                // No text selected. Let's check if there's a video.
                chrome.tabs.sendMessage(tab.id, { action: "getVideoCapability" }, (vidResponse) => {
                    if (chrome.runtime.lastError || !vidResponse || !vidResponse.videoDetected) {
                        showState('idle');
                        return;
                    }

                    currentCapability = vidResponse;

                    // YouTube Shorts: no transcript possible, route directly to visual scan
                    if (vidResponse.isShorts) {
                        const titleEl = document.getElementById('no-transcript-title');
                        const reasonEl = document.getElementById('no-transcript-reason');
                        if (titleEl) titleEl.textContent = '📱 YouTube Short Detected';
                        if (reasonEl) reasonEl.textContent = 'YouTube Shorts do not support transcripts. Scan visually for deepfakes instead.';
                        showState('videoNoTranscript');
                        return;
                    }
                    
                    if (vidResponse.transcript && (vidResponse.transcript.status === 'available' || vidResponse.transcript.status === 'captions_only')) {
                        if (vidResponse.transcript.isAutoGenerated) {
                            ui.videoTranscriptInfo.textContent = "Based on auto-generated captions.";
                        } else {
                            ui.videoTranscriptInfo.textContent = "Accessible transcript available.";
                        }
                        
                        // Check if job already running
                        fetch(`http://127.0.0.1:8000/api/video/analyze/video/${currentCapability.videoId}`)
                        .then(r => {
                            if (r.ok) return r.json();
                            throw new Error("Job not found");
                        }).then(job => {
                            startPolling(job.job_id);
                        }).catch(() => {
                            // No active job found, show ready state
                            showState('videoReady');
                        });
                        
                    } else {
                        showState('videoNoTranscript');
                    }
                });
            }
        });
    } catch (err) {
        console.error("Failed to read selection:", err);
        showState('idle');
    }

    // 1.5 Handle Video Analysis
    if (ui.analyzeVideoBtn) {
        ui.analyzeVideoBtn.addEventListener('click', async () => {
            if (!currentCapability || isAnalyzingVideo) return;
            isAnalyzingVideo = true;
            
            showState('analyzingVideo');
            
            let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            
            chrome.tabs.sendMessage(tab.id, { 
                action: "acquireTranscript", 
                capability: currentCapability
            }, (response) => {
                isAnalyzingVideo = false;
                
                if (chrome.runtime.lastError || !response) {
                    ui.errorMessage.textContent = "Failed to communicate with page.";
                    showState('error');
                    return;
                }
                
                if (response.status === "error") {
                    ui.errorMessage.textContent = response.message || "Failed to retrieve transcript.";
                    showState('error');
                } else if (response.status === "success" && response.transcript) {
                    const segCount = response.transcript.segments ? response.transcript.segments.length : 0;
                    ui.transcriptStatsInfo.textContent = `${segCount} transcript segments`;
                    ui.transcriptAutoInfo.textContent = response.transcript.isAutoGenerated 
                        ? "Based on auto-generated captions" 
                        : "Based on native transcript";
                    showState('transcriptReady');
                }
            });
        });
    }
    
    function renderJob(job) {
        ui.videoJobStats.textContent = `${job.completed_claims} / ${job.total_claims} verified`;
        ui.videoJobClaims.innerHTML = '';
        
        job.results.forEach(res => {
            const div = document.createElement('div');
            div.style.padding = '8px';
            div.style.background = 'rgba(255,255,255,0.05)';
            div.style.borderRadius = '4px';
            div.style.fontSize = '12px';
            
            let statusIcon = '◌ Checking...';
            let color = '#94a3b8';
            if (res.status === 'verified') {
                if (res.verdict === 'true') {
                    statusIcon = '✓ TRUE';
                    color = '#10b981';
                } else if (res.verdict === 'false') {
                    statusIcon = '✓ FALSE';
                    color = '#ef4444';
                } else if (res.verdict === 'misleading') {
                    statusIcon = '✓ MISLEADING';
                    color = '#f59e0b';
                } else {
                    statusIcon = '✓ ' + res.verdict.toUpperCase();
                    color = '#f59e0b';
                }
            } else if (res.status === 'unverifiable') {
                statusIcon = '○ Insufficient evidence';
            } else if (res.status === 'error') {
                statusIcon = '× Unable to verify — Retry';
                color = '#ef4444';
            } else if (res.status === 'pending') {
                statusIcon = '○ Waiting...';
            }
            
            const mins = Math.floor(res.start_time / 60);
            const secs = Math.floor(res.start_time % 60).toString().padStart(2, '0');
            
            div.innerHTML = `
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px; color: ${color}; font-weight: bold;">
                    <span>${statusIcon}</span>
                    <span>${mins}:${secs}</span>
                </div>
                <div style="color: #cbd5e1; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; line-height: 1.4;">
                    ${res.text}
                </div>
            `;
            ui.videoJobClaims.appendChild(div);
        });
    }

    function startPolling(job_id) {
        showState('videoJob');
        if (pollInterval) clearInterval(pollInterval);
        
        const poll = async () => {
            try {
                const statusResp = await fetch(`http://127.0.0.1:8000/api/video/analyze/${job_id}`);
                if (statusResp.ok) {
                    const job = await statusResp.json();
                    renderJob(job);
                    if (job.status === 'completed' || job.status === 'failed') {
                        clearInterval(pollInterval);
                    }
                }
            } catch (e) {
                console.error("Polling error", e);
            }
        };
        
        poll();
        pollInterval = setInterval(poll, 2000);
    }

    if (ui.continueVideoBtn) {
        ui.continueVideoBtn.addEventListener('click', async () => {
            if (!currentCapability) return;
            showState('videoJob');
            
            try {
                // Phase 5.4: Generate claims
                const claimsResponse = await fetch('http://127.0.0.1:8000/api/video/claims', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ video_id: currentCapability.videoId })
                });
                
                if (!claimsResponse.ok) throw new Error("Failed to extract claims");
                
                // Phase 5.5: Start progressive analysis
                const response = await fetch('http://127.0.0.1:8000/api/video/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ video_id: currentCapability.videoId })
                });
                if (!response.ok) throw new Error("Failed to start analysis");
                
                const data = await response.json();
                startPolling(data.job_id);
            } catch (err) {
                ui.errorMessage.textContent = err.message || "Failed to start video analysis.";
                showState('error');
            }
        });
    }

    // 2. Handle Verification
    ui.verifyBtn.addEventListener('click', async () => {
        if (!selectedClaim) return;
        
        showState('verifying');

        try {
            // Respecting the actual backend pipeline logic
            const response = await fetch('http://127.0.0.1:8000/api/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: selectedClaim })
            });

            if (!response.ok) {
                throw new Error(`Server returned ${response.status}`);
            }

            /** @type {VerifyResponse} */
            const data = await response.json();
            
            renderResult(data);
            showState('result');
            
        } catch (err) {
            console.error("Verification error:", err);
            ui.errorMessage.textContent = "Failed to connect to verification server. Ensure the TruthLens backend is running.";
            showState('error');
        }
    });

    // 3. Render Result Logic
    function renderResult(data) {
        // Reset classes
        ui.verdictBadge.className = 'tl-verdict-badge';
        
        // Display the claim
        const displayClaim = selectedClaim.length > 200 
            ? selectedClaim.substring(0, 200) + "..." 
            : selectedClaim;
        ui.resultClaimText.textContent = `"${displayClaim}"`;

        const verdict = data.verdict;
        const confidenceStr = Math.round(data.confidence_score * 100) + "% confidence";
        ui.confidenceScore.textContent = confidenceStr;

        // Semantic mapping
        if (verdict === 'true') {
            ui.verdictBadge.textContent = "True";
            ui.verdictBadge.classList.add('verdict-true');
        } else if (verdict === 'false') {
            ui.verdictBadge.textContent = "False";
            ui.verdictBadge.classList.add('verdict-false');
        } else if (verdict === 'misleading') {
            ui.verdictBadge.textContent = "Misleading";
            ui.verdictBadge.classList.add('verdict-misleading');
        } else {
            ui.verdictBadge.textContent = "Unverifiable";
            ui.verdictBadge.classList.add('verdict-unverifiable');
            // Hide confidence for unverifiable since it's 0.0
            ui.confidenceScore.textContent = "Opinion or insufficient evidence";
        }

        // Render sources
        ui.sourcesList.innerHTML = '';
        if (data.sources && data.sources.length > 0) {
            ui.sourcesContainer.classList.remove('hidden');
            ui.unverifiableExplainer.classList.add('hidden');
            
            data.sources.forEach(source => {
                const a = document.createElement('a');
                a.className = 'tl-source-card';
                a.href = source.url;
                a.target = '_blank';
                a.rel = 'noopener noreferrer';
                
                const publisherDiv = document.createElement('div');
                publisherDiv.className = 'tl-source-publisher';
                publisherDiv.innerHTML = `
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                    ${source.publisher || new URL(source.url).hostname}
                `;
                
                const titleDiv = document.createElement('div');
                titleDiv.className = 'tl-source-title';
                titleDiv.textContent = source.title;
                
                a.appendChild(publisherDiv);
                a.appendChild(titleDiv);
                ui.sourcesList.appendChild(a);
            });
        } else {
            ui.sourcesContainer.classList.add('hidden');
            ui.unverifiableExplainer.classList.remove('hidden');
        }
    }

    // 4. Utility Buttons
    ui.resetBtn.addEventListener('click', () => {
        window.close();
        showState(currentCapability ? 'videoReady' : 'idle');
    });

    ui.retryBtn.addEventListener('click', () => {
        showState('ready'); // Go back to ready to try again
    });

    // 5. Restore Legacy Feature: Highlight Loaded Language
    const explainBtn = document.getElementById('explain-btn');
    if (explainBtn) {
        explainBtn.addEventListener('click', async () => {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (!tab.url.startsWith('chrome://')) {
                chrome.tabs.sendMessage(tab.id, { action: "triggerExplainer" });
                window.close(); // Close the popup so the user can see the highlights
            }
        });
    }

    // 6. Visual Fake Video Analysis (No Audio / Deepfake Detection)
    async function runVisualVideoScan() {
        showState('analyzingVideo');
        if (ui.analyzingVideoTitle) ui.analyzingVideoTitle.textContent = "Scanning Video Frames...";
        if (ui.analyzingVideoSubtitle) ui.analyzingVideoSubtitle.textContent = "Sampling frames across video duration (no audio required)...";

        let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tab || !tab.id) {
            ui.errorMessage.textContent = "Could not find active browser tab.";
            showState('error');
            return;
        }

        chrome.tabs.sendMessage(tab.id, { action: "captureVideoFrames", maxFrames: 12 }, async (response) => {
            if (chrome.runtime.lastError || !response) {
                ui.errorMessage.textContent = "Could not connect to the video player on this page. Try opening the video directly or use local video scan.";
                showState('error');
                return;
            }

            if (response.status === "error" || !response.frames || response.frames.length === 0) {
                ui.errorMessage.textContent = response.message || "Could not extract video frames.";
                showState('error');
                return;
            }

            if (ui.analyzingVideoTitle) ui.analyzingVideoTitle.textContent = "Analyzing Visual Authenticity...";
            if (ui.analyzingVideoSubtitle) ui.analyzingVideoSubtitle.textContent = `Processing ${response.frames.length} frames for deepfake manipulation...`;

            try {
                const apiRes = await fetch("http://127.0.0.1:8000/api/video/analyze-visual-frames", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        video_id: currentCapability?.videoId || "page_video",
                        title: tab.title || "Web Video",
                        frames: response.frames
                    })
                });

                if (!apiRes.ok) {
                    const errData = await apiRes.json().catch(() => ({}));
                    throw new Error(errData.detail || `Server returned error ${apiRes.status}`);
                }

                const data = await apiRes.json();
                renderVisualResult(data);
                showState('visualResult');
            } catch (err) {
                console.error("Visual analysis error:", err);
                ui.errorMessage.textContent = err.message || "Failed to complete visual video scan.";
                showState('error');
            }
        });
    }

    function renderVisualResult(data) {
        ui.visualVerdictBadge.className = 'tl-verdict-badge';
        const verdict = data.verdict;
        ui.visualVerdictBadge.textContent = verdict;
        if (verdict === 'DEEPFAKE') {
            ui.visualVerdictBadge.classList.add('verdict-deepfake');
        } else if (verdict === 'SUSPICIOUS') {
            ui.visualVerdictBadge.classList.add('verdict-suspicious');
        } else {
            ui.visualVerdictBadge.classList.add('verdict-authentic');
        }

        ui.visualConfidenceScore.textContent = `${Math.round(data.confidence_score * 100)}% confidence`;

        if (data.flagged_segments && data.flagged_segments.length > 0) {
            ui.visualFlaggedBanner.classList.remove('hidden');
            ui.visualFlaggedTimes.textContent = data.flagged_segments.join(', ');
        } else {
            ui.visualFlaggedBanner.classList.add('hidden');
        }

        const fb = data.forensic_breakdown || {};
        ui.riskFace.textContent = `${Math.round((fb.face_deepfake_risk || 0) * 100)}%`;
        if (ui.riskSeam) ui.riskSeam.textContent = `${Math.round((fb.boundary_seam_risk || 0) * 100)}%`;
        ui.riskScene.textContent = `${Math.round((fb.scene_ai_generation_risk || 0) * 100)}%`;
        ui.riskFft.textContent = `${Math.round((fb.frequency_artifacts_risk || 0) * 100)}%`;
        ui.riskTemporal.textContent = `${Math.round((fb.temporal_inconsistency_risk || 0) * 100)}%`;

        const riskEls = [ui.riskFace, ui.riskScene, ui.riskFft, ui.riskTemporal];
        if (ui.riskSeam) riskEls.push(ui.riskSeam);

        riskEls.forEach(el => {
            const val = parseInt(el.textContent, 10) || 0;
            if (val >= 55) el.style.color = '#ef4444';
            else if (val >= 35) el.style.color = '#f59e0b';
            else el.style.color = '#10b981';
        });

        ui.visualFrameCount.textContent = `${data.analyzed_frames_count} frames analyzed`;
        ui.visualTimelineList.innerHTML = '';

        if (data.timeline && data.timeline.length > 0) {
            data.timeline.forEach(item => {
                const div = document.createElement('div');
                div.style.padding = '6px 8px';
                div.style.background = item.flagged ? 'rgba(239, 68, 68, 0.08)' : 'var(--bg-secondary)';
                div.style.border = item.flagged ? '1px solid rgba(239, 68, 68, 0.3)' : '1px solid var(--border-color)';
                div.style.borderRadius = '4px';
                div.style.fontSize = '11px';

                const statusColor = item.flagged ? '#ef4444' : '#10b981';
                const statusLabel = item.flagged ? 'FLAGGED' : 'NORMAL';
                const faceText = item.face_detected ? `Face detected (${item.face_count})` : 'Full Scene';

                div.innerHTML = `
                    <div style="display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 2px;">
                        <span style="color: var(--text-primary); font-family: monospace;">⏱ ${item.formatted_time}</span>
                        <span style="color: ${statusColor}; font-size: 10px;">${statusLabel} (${Math.round(item.fake_score * 100)}%)</span>
                    </div>
                    <div style="color: var(--text-secondary); font-size: 10px; display: flex; justify-content: space-between;">
                        <span>${faceText}</span>
                        <span>${item.signals && item.signals.length > 0 ? item.signals[0].replace(/_/g, ' ') : 'natural'}</span>
                    </div>
                `;
                ui.visualTimelineList.appendChild(div);
            });
        }

        ui.visualExplanation.textContent = data.explanation || "Visual forensic scan completed.";
    }

    if (ui.analyzeVisualBtn) {
        ui.analyzeVisualBtn.addEventListener('click', runVisualVideoScan);
    }
    if (ui.analyzeVisualReadyBtn) {
        ui.analyzeVisualReadyBtn.addEventListener('click', runVisualVideoScan);
    }
    if (ui.visualResetBtn) {
        ui.visualResetBtn.addEventListener('click', () => {
            window.close();
            showState(currentCapability ? 'videoReady' : 'idle');
        });
    }

    const visualBackBtn = document.getElementById('visual-back-btn');
    if (visualBackBtn) {
        visualBackBtn.addEventListener('click', () => {
            const isNoTranscript = currentCapability && (!currentCapability.transcript || currentCapability.transcript.status !== 'available');
            showState(currentCapability ? (isNoTranscript ? 'videoNoTranscript' : 'videoReady') : 'idle');
        });
    }

    const claimBackBtn = document.getElementById('claim-back-btn');
    if (claimBackBtn) {
        claimBackBtn.addEventListener('click', () => {
            showState(currentCapability ? 'videoReady' : 'idle');
        });
    }

    const headerCloseBtn = document.getElementById('header-close-btn');
    if (headerCloseBtn) {
        headerCloseBtn.addEventListener('click', () => {
            window.close();
        });
    }

    if (ui.localVideoFile) {
        ui.localVideoFile.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            showState('analyzingVideo');
            if (ui.analyzingVideoTitle) ui.analyzingVideoTitle.textContent = "Analyzing Video File...";
            if (ui.analyzingVideoSubtitle) ui.analyzingVideoSubtitle.textContent = `Scanning ${file.name} for deepfake patterns...`;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch("http://127.0.0.1:8000/api/video/analyze-visual-file", {
                    method: "POST",
                    body: formData
                });

                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || "Video upload analysis failed");
                }

                const data = await res.json();
                renderVisualResult(data);
                showState('visualResult');
            } catch (err) {
                console.error("Local video error:", err);
                ui.errorMessage.textContent = err.message || "Failed to analyze video file.";
                showState('error');
            }
        });
    }
});