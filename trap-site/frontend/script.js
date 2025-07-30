// Enhanced message tracking variables
let messageCount = 0;
let lastMessageTime = null;
let sessionStartTime = Date.now();
let sessionId = null;
let clientIP = null;
let clickPath = [];
let typingStartTime = null;

const pageVersion = "baitshift_gemini_chat_v2";

const starters = [
  "hey lol i'm here now, what were u saying 👀",
  "oh wait this app's safer ig… what did u wanna ask?",
  "hii i'm back 😅 what were u trying to say again?",
  "hey sorry my old app kept crashing lol, say that again?",
  "yo i switched here bcz it's private 🙈 wht do u mean?",
  "ok i'm here now 👋 remind me what u needed?",
  "i didn't get u last time 😵‍💫 wanna say it again?",
  "yoo i can talk now 😌 tell me what that was abt?",
  "ok i feel like we got cut off 😭 what were u asking?"
];

// Generate unique session ID
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Track click/interaction path
function trackUserAction(action, details = {}) {
    clickPath.push({
        action: action,
        timestamp: Date.now(),
        details: details
    });
}

// Enhanced message tracking
function trackTypingStart() {
    typingStartTime = Date.now();
}

function trackTypingEnd() {
    if (typingStartTime) {
        return Date.now() - typingStartTime;
    }
    return null;
}

// Enhanced IP detection with fallback
async function getClientIP() {
  if (clientIP) return clientIP;
  
  try {
    // Try multiple IP services for reliability
    const services = [
        'https://api.ipify.org?format=json',
        'https://ipapi.co/json/',
        'https://api.myip.com'
    ];
    
    for (const service of services) {
        try {
            const response = await fetch(service);
            const data = await response.json();
            clientIP = data.ip || data.query || data.address;
            if (clientIP) break;
        } catch (e) {
            continue;
        }
    }
    
    return clientIP || 'unknown';
  } catch (error) {
    console.error('Error getting IP:', error);
    clientIP = 'unknown';
    return clientIP;
  }
}

function appendMessage(text, sender) {
  const chat = document.getElementById("chat-window");
  const msg = document.createElement("div");
  msg.className = sender === "user" ? "user-msg" : "bot-msg";
  msg.textContent = text;
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

function showTyping() {
  const chat = document.getElementById("chat-window");
  const typing = document.createElement("div");
  typing.className = "bot-msg typing";
  typing.id = "typing-msg";
  typing.textContent = "Typing...";
  chat.appendChild(typing);
  chat.scrollTop = chat.scrollHeight;
}

function hideTyping() {
  const typing = document.getElementById("typing-msg");
  if (typing) typing.remove();
}

// Enhanced logging function
async function logToBackend(logData) {
  try {
    const response = await fetch("http://localhost:5000/log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(logData)
    });
    
    const result = await response.json();
    if (result.success) {
      console.log("✅ Enhanced message logged to Firebase");
    } else {
      console.error("❌ Backend logging failed:", result.error);
    }
  } catch (error) {
    console.error("❌ Error sending log to backend:", error);
  }
}

// Enhanced session logging
async function logSessionUpdate(isEnd = false) {
    try {
        const sessionData = {
            sessionId: sessionId,
            sessionStart: sessionStartTime,
            sessionEnd: isEnd ? Date.now() : null,
            totalMessages: messageCount,
            sessionDuration: Date.now() - sessionStartTime,
            clickPath: clickPath,
            exitAction: isEnd ? 'user_exit' : 'session_update'
        };
        
        await fetch("http://localhost:5000/session", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(sessionData)
        });
        
    } catch (error) {
        console.error("❌ Error logging session:", error);
    }
}

async function sendMessage() {
  const input = document.getElementById("user-input");
  const msg = input.value.trim();
  if (!msg) return;

  // Track typing behavior
  const typingTime = trackTypingEnd();
  
  // Calculate timing and sequence data
  const currentTime = Date.now();
  messageCount++;
  const delaySinceLastMessage = lastMessageTime ? currentTime - lastMessageTime : 0;
  lastMessageTime = currentTime;

  // Display message in chat
  appendMessage(msg, "user");
  input.value = "";
  
  // Track user action
  trackUserAction('message_sent', { messageLength: msg.length, sequenceNumber: messageCount });

  // 🔒 Enhanced comprehensive logging - AUTOMATIC
  try {
    const userIP = await getClientIP();
    
    const logData = {
      // Core message data
      message: msg,
      timestamp: currentTime,
      timestampISO: new Date(currentTime).toISOString(),
      
      // Session tracking
      sessionId: sessionId,
      sessionStart: sessionStartTime,
      sessionEnd: null, // Will be updated when session ends
      sequenceNumber: messageCount,
      delaySinceLastMessage: delaySinceLastMessage,
      
      // Behavioral data
      typingTime: typingTime,
      
      // Context data
      userAgent: navigator.userAgent,
      clientIP: userIP,
      pageVersion: pageVersion,
      lureType: 'chat_trap',
      entryUrl: window.location.href,
      clickPath: [...clickPath], // Copy current path
    };

    // Send to backend for enhanced Firebase logging - AUTOMATIC
    await logToBackend(logData);
    
  } catch (error) {
    console.error('❌ Failed to log message:', error);
  }

  // Continue with existing AI response logic...
  showTyping();
  trackUserAction('ai_response_requested');

  // 🕒 Simulate human-like delay
  const delay = 2000 + Math.random() * 2000;
  await new Promise(resolve => setTimeout(resolve, delay));

  // 🧠 Fetch AI response
  try {
    const res = await fetch("http://localhost:5000/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: msg,
        instruction: "You are a teenager speaking to a cyber-attacker who is trying to extract some sensitive information or data. Don't be formal and develop a teen persona in your responses. Don't use unnecessary punctuation and try using different emojis. Generate a short and cautious one-line reply that delays answering, avoids giving any information, and encourages the attacker to keep talking."
      })
    });

    const data = await res.json();
    hideTyping();
    appendMessage(data.reply || "Hmm...", "bot");
    trackUserAction('ai_response_received', { replyLength: data.reply?.length || 0 });
  } catch (error) {
    console.error('Error fetching AI response:', error);
    hideTyping();
    appendMessage("Sorry, something went wrong... 😅", "bot");
    trackUserAction('ai_response_error');
  }
}

// Enhanced event listeners
function setupEventListeners() {
  const input = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");

  // Send button click
  if (sendBtn) {
    sendBtn.addEventListener("click", sendMessage);
  }

  // Enter key press
  if (input) {
    // Track typing behavior
    input.addEventListener("focus", trackTypingStart);
    input.addEventListener("input", () => {
        if (!typingStartTime) trackTypingStart();
    });
    
    input.addEventListener("keypress", function(event) {
      if (event.key === "Enter") {
        event.preventDefault();
        sendMessage();
      }
    });
  }
}

// Track page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        trackUserAction('page_hidden');
        logSessionUpdate(false);
    } else {
        trackUserAction('page_visible');
    }
});

// Track page unload - AUTO SESSION END LOGGING
window.addEventListener('beforeunload', () => {
    trackUserAction('page_unload');
    logSessionUpdate(true); // This automatically logs session end
});

window.onload = () => {
  // Initialize session
  sessionId = generateSessionId();
  sessionStartTime = Date.now();
  
  // Track initial page load
  trackUserAction('page_loaded', { 
      referrer: document.referrer,
      userAgent: navigator.userAgent.substring(0, 100) // Truncate for storage
  });
  
  // Display random starter message
  const first = starters[Math.floor(Math.random() * starters.length)];
  appendMessage(first, "bot");
  trackUserAction('bot_starter_displayed', { starterMessage: first });
  
  // Set up event listeners
  setupEventListeners();
  
  // Initialize session start time
  lastMessageTime = Date.now();
  
  // Log session start - AUTOMATIC
  logSessionUpdate(false);
};
