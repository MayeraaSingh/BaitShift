chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "baitshiftScan",
    title: "Scan this text with BaitShift",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info) => {
  if (info.menuItemId === "baitshiftScan" && info.selectionText) {
    chrome.storage.local.set({ lastText: info.selectionText });
  }
});

// Handle clicks on the extension icon
chrome.action.onClicked.addListener((tab) => {
  // Send a message to content script to show panel with last analyzed text
  chrome.tabs.sendMessage(tab.id, { action: "showLastAnalysis" });
});
