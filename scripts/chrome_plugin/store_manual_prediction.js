// ContentScript
(function() {

    // chrome.runtime.onMessage.addListener(
    //     function(request, sender, sendResponse) {
    //         //console.log(request.payload + "called inside contentscript")
    //         input_text = request.payload
    //     }
    // );

    // I think we have to convert the other predictions to this so we can access the selected text here
    chrome.storage.local.get('class_payload', function (items) {
        // console.log(items.class_payload)
        window.my_class = items.class_payload
        //chrome.storage.local.remove('payload'); // removes the given key value
    });

    chrome.storage.local.get('text_payload', function (items) {
        // console.log(items.text_payload);
        window.text = items.text_payload;
        //chrome.storage.local.remove('text_payload'); // removes the given key value
    });

    console.log(text, my_class)
})();
