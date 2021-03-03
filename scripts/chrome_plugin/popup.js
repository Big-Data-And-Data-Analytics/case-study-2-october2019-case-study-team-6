document.addEventListener('DOMContentLoaded', function() {
    var button = document.getElementById('theButton');

    button.onclick = alertSelection;
    function alertSelection() {
        {
            chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
                chrome.tabs.executeScript(tabs[0].id, {
                    file: 'predict_id_motive.js'
                    //code: 'alert(window.getSelection().toString());'
                });
            });
        };
    }
}, false);
