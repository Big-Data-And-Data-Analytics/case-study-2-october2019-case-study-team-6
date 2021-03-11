// popup.js
document.addEventListener('DOMContentLoaded', function() {
    var id_button = document.getElementById('id_button');
    var ni_button = document.getElementById('ni_button');
    var manual_predict_button = document.getElementById('manual_predict_button');
    var class_dropdown = document.getElementById('class_selector');
    var country_dropdown = document.getElementById('country_selector');

    id_button.onclick = predict_id_motive;
    function predict_id_motive() {
        {
            chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
                chrome.tabs.executeScript(tabs[0].id, {
                    file: 'predict_id_motive.js'
                });

                chrome.storage.local.get('response_payload_im', function (items) {
                    my_class = JSON.parse(items.response_payload_im);
                    my_class = my_class.prediction
                    // my_class = my_class.replace('{"prediction":"', "");
                    // my_class = my_class.replace('"}', '');
                    document.getElementById("output_id").innerHTML = my_class;
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
                    my_class = JSON.parse(items.response_payload_ni);
                    my_class = my_class.prediction
                    document.getElementById("output_in").innerHTML = my_class;
                    document.getElementById("country_selector").value = my_class;
                });
            });
        };
    }

    manual_predict_button.onclick = manual_predict;
    function manual_predict() {
        {
            chrome.storage.local.set({
                class_payload : class_dropdown.value,
                country_payload : country_dropdown.value
            }, function () {
                chrome.tabs.executeScript({
                    file: "store_manual_prediction.js"
                });
            });
        };
    }

}, false);
