chrome.storage.onChanged.addListener((changes) => {
  if (changes.lastText && changes.lastText.newValue) {
    showPanel(changes.lastText.newValue);
  }
});

// Listen for messages from the background script
chrome.runtime.onMessage.addListener((message) => {
  if (message.action === "showLastAnalysis") {
    // Get the last analyzed text from storage and show panel
    chrome.storage.local.get("lastText", (result) => {
      if (result.lastText) {
        showPanel(result.lastText);
      } else {
        // No previous analysis, show a notification panel
        showNotificationPanel("No message has been analyzed yet. Select text on the page and use the context menu to analyze it.");
      }
    });
  }
});

// Function to show a simple notification panel
function showNotificationPanel(message) {
  const oldPanel = document.getElementById("baitshift-panel");
  if (oldPanel) oldPanel.remove();
  
  const panel = document.createElement("div");
  panel.id = "baitshift-panel";
  panel.style = `
    position: fixed;
    bottom: 140px;
    right: 40px;
    width: 300px;
    background: #f8f4ff;
    color: #333;
    border: 1px solid #e2d5f7;
    border-radius: 14px;
    padding: 16px;
    z-index: 10000;
    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    font-family: 'Segoe UI', sans-serif;
    font-size: 14px;
  `;
  
  panel.innerHTML = `
    <div style="margin:10px 0; padding:8px 12px; background:#e0e7ff; border-radius:8px; font-size:14.5px; line-height:1.4;"><span style="margin-right:5px;">⚠️</span> ${message}</div>
    <div style="display:flex; justify-content:flex-end; margin:10px 0;">
      <button id="baitshift-close" style="background:#c7b2e2; color:#333; border:none; padding:8px 12px; border-radius:8px; cursor:pointer; font-size:13px; transition:all 0.2s ease;">Close</button>
    </div>
  `;
  
  document.body.appendChild(panel);
  
  panel.querySelector("#baitshift-close").addEventListener("click", () => {
    panel.remove();
  });
}

// Function to create + show the panel
function showPanel(selectedText) {
  // If panel already exists → remove and create fresh (force refresh)
  const oldPanel = document.getElementById("baitshift-panel");
  if (oldPanel) oldPanel.remove();

  const panel = document.createElement("div");
  panel.id = "baitshift-panel";
  panel.style = `
    position: fixed;
    bottom: 140px;
    right: 40px;
    width: 300px;
    background: #f8f4ff;  /* lighter lilac background */
    color: #333;
    border: 1px solid #e2d5f7;
    border-radius: 14px;
    padding: 16px;
    z-index: 10000;
    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    font-family: 'Segoe UI', sans-serif;
    font-size: 14px;
  `;
  panel.innerHTML = `
    <div style="font-weight:600; margin-bottom:12px; font-size:15px; display:flex; align-items:center;">
      <span style="margin-right:6px;">🧿</span> ${selectedText}
    </div>
    <div id="baitshift-risk" style="margin:10px 0; padding:8px 12px; background:#e0e7ff; border-radius:8px; font-size:14.5px; font-weight:500;">Risk Level: Analyzing...</div>
    <div id="baitshift-reply" style="margin:12px 0; padding:8px 12px; background:#f0f9ff; border-radius:8px; font-size:14.5px; line-height:1.4; max-height:80px; overflow-y:auto;">Generating reply...</div>
    <div style="display:flex; justify-content:space-between; margin:10px 0;">
      <button id="baitshift-copy" style="background:#c7b2e2; color:#333; border:none; padding:8px 14px; border-radius:8px; cursor:pointer; font-weight:500; transition:all 0.2s ease;">Copy Reply</button>
      <button id="baitshift-close" style="background:#c7b2e2; color:#333; border:none; padding:8px 12px; border-radius:8px; cursor:pointer; font-size:13px; transition:all 0.2s ease;">Close</button>
    </div>
    <div id="baitshift-instructions" style="margin:10px 0 0 0; padding:8px 10px; background:#f5f5f5; border-radius:6px; font-size:12px; color:#777; line-height:1.3;"><span style="margin-right:5px;">⚠️</span> Getting instructions...</div>
  `;

  document.body.appendChild(panel);

  // Replace dummy risk score with NLP analysis
  analyzeMessage(selectedText);

  // Copy reply button (will be updated with real reply)
  panel.querySelector("#baitshift-copy").addEventListener("click", () => {
    const reply = window.baitshiftReply || "This is your bait reply!";
    const textarea = document.createElement("textarea");
    textarea.value = reply;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
    alert("✅ Reply copied!");
  });

  // Close button
  panel.querySelector("#baitshift-close").addEventListener("click", () => {
    panel.remove();
  });
}

// OPTIONAL: on first load → if already have lastText, show panel
chrome.storage.local.get("lastText", (result) => {
  if (result.lastText) {
    showPanel(result.lastText);
  }
});

// New function to call NLP analysis
async function analyzeMessage(message) {
  try {
    console.log('🔍 Analyzing message:', message);
    
    const response = await fetch('http://localhost:5001/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: message })
    });

    const result = await response.json();
    
    // Debug: Print to console as requested
    console.log('--- NLP ANALYSIS ---');
    console.log('Risk Level:', result.risk_level);
    console.log('Reply:', result.reply);
    console.log('Category:', result.category);
    console.log('Instructions:', result.user_instructions);
    
    // Update UI elements directly (no delay!)
    const riskElement = document.getElementById("baitshift-risk");
    const instructionsElement = document.getElementById("baitshift-instructions");
    const replyElement = document.getElementById("baitshift-reply");
    
    if (riskElement) {
      const riskColor = result.risk_level === 'High' ? '#fecaca' : 
                       result.risk_level === 'Medium' ? '#fed7aa' : '#d1fae5';
      riskElement.style.background = riskColor;
      riskElement.textContent = `Risk Level: ${result.risk_level}`;
    }
    
    if (instructionsElement) {
      instructionsElement.innerHTML = `<span style="margin-right:5px;">⚠️</span> ${result.user_instructions}`;
    }
    
    if (replyElement) {
      replyElement.textContent = `AI Reply: ${result.reply}`;
    }
    
    // Store reply for copy button
    window.baitshiftReply = result.reply;
    
  } catch (error) {
    console.error('❌ Analysis failed:', error);
    
    // Update UI with error state
    const riskElement = document.getElementById("baitshift-risk");
    const instructionsElement = document.getElementById("baitshift-instructions");
    const replyElement = document.getElementById("baitshift-reply");
    
    if (riskElement) {
      riskElement.textContent = "Risk Level: Connection Error";
      riskElement.style.background = "#fee2e2";
    }
    if (instructionsElement) {
      instructionsElement.innerHTML = `<span style="margin-right:5px;">⚠️</span> Unable to get instructions. Please check if the server is running.`;
    }
    if (replyElement) {
      replyElement.textContent = "AI Reply: Service unavailable";
      replyElement.style.background = "#f3f4f6";
    }
  }
}
