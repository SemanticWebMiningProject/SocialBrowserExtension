chrome.runtime.onInstalled.addListener(function() {
    chrome.history.search({
        'text': '',
        'maxResults': 75
    }, function(historyItem) {
        console.log(historyItem)
    });
});

chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
        if (request.sendToDB == true)
            console.log("Response sent")
            sendResponse({response: ("sendToDB reads TRUE")});
    }
);