document.addEventListener('DOMContentLoaded', function() {
    var id_button = document.getElementById('id_button');
    var ni_button = document.getElementById('ni_button');
    var test_button = document.getElementById('test_button');

    id_button.onclick = predict_id_motive;
    function predict_id_motive() {
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
        {
            chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
                chrome.tabs.executeScript(tabs[0].id, {
                    file: 'predict_ni.js'
                });
            });
        };
    }

    test_button.onclick = test;
    function test() {
        {
            chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
                chrome.tabs.executeScript(tabs[0].id, {
                    file: 'output_display_test.js'
                });

                chrome.runtime.onMessage.addListener(
                    function(request, sender, sendResponse) {
                        document.getElementById("extensionpopupcontent").innerHTML = request.payload;
                    }
                );
            });
        };
    }

}, false);
