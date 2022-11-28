let sendGraphButton = document.getElementById('changeColor');
console.log(sendGraphButton)

sendGraphButton.onclick = function(element) {

    chrome.runtime.sendMessage({sendToDB: true}, function(response) {
        console.log(response)
    });

    // put history data in sync
};