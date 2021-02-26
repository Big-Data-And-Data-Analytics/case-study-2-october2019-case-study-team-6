document.addEventListener('DOMContentLoaded', function() {
    var button = document.getElementById('theButton');

    button.onclick = alertSelection;
    function alertSelection() {
        {
            chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
                chrome.tabs.executeScript(tabs[0].id, {
                    file: 'inject.js'
                    //code: 'alert(window.getSelection().toString());'
                });
            });
        };
    }
}, false);
