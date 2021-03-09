// popup.js
document.addEventListener('DOMContentLoaded', function() {
    var id_button = document.getElementById('id_button');
    var ni_button = document.getElementById('ni_button');
    var manual_predict_button = document.getElementById('manual_predict_button');
    var class_dropdown = document.getElementById('class_selector');

    id_button.onclick = predict_id_motive;
    function predict_id_motive() {
        {
            chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
                chrome.tabs.executeScript(tabs[0].id, {
                    file: 'predict_id_motive.js'
                });

                chrome.storage.local.get('response_payload_im', function (items) {
                    document.getElementById("extensionpopupcontent").innerHTML = items.response_payload_im;
                    // Not sure if the response is of type string, might have to convert it from JSON to string
                    my_class = items.response_payload_im;
                    my_class = my_class.replace('{"prediction":"', "");
                    my_class = my_class.replace('"}', '');
                    document.getElementById("class_selector").value = my_class;
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

                chrome.storage.local.get('response_payload_ni', function (items) {
                    document.getElementById("extensionpopupcontent").innerHTML = items.response_payload_ni;
                });
            });
        };
    }

    manual_predict_button.onclick = manual_predict;
    function manual_predict() {
        {
            input_class = class_dropdown.value

            chrome.storage.local.set({
                class_payload : class_dropdown.value
            }, function () {
                chrome.tabs.executeScript({
                    file: "store_manual_prediction.js"
                });
            });
        };
    }

}, false);
