document.addEventListener('DOMContentLoaded', function() {
    var id_button = document.getElementById('id_button');
    var ni_button = document.getElementById('ni_button');

    id_button.onclick = predict_id_motive;
    function predict_id_motive() {
        alert("Predicting id...")
        {
            chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
                chrome.tabs.executeScript(tabs[0].id, {
                    file: 'predict_id_motive.js'
                });
            });
        };
    }

    ni_button.onclick = predict_ni;
    function predict_ni() {
        alert("Predicting ni...")
        {
            chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
                chrome.tabs.executeScript(tabs[0].id, {
                    file: 'predict_ni.js'
                });
            });
        };
    }
}, false);
