// ContentScript
(function() {

    chrome.storage.local.get('class_payload', function (items) {
        // console.log(items.class_payload)
        const my_class = items.class_payload
        chrome.storage.local.remove('class_payload'); // removes the given key value
        
        chrome.storage.local.get('text_payload', function (items) {
            // console.log(items.text_payload);
            const text = items.text_payload;
            console.log(text, my_class)
            chrome.storage.local.remove('text_payload'); // removes the given key value

            const req = new XMLHttpRequest();
            const baseUrl = "http://localhost:5000/ENDPOINT"; // Endpoint for storing to DB
        
            var data = {sentence: text, identity_motive: my_class};
            req.open(method="POST", url=baseUrl, true);
            req.setRequestHeader("Content-Type", "application/json");
            req.send(JSON.stringify(data));
            req.onreadystatechange = function() { // Call a function when the state changes.
                if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
                    // Also something could be here for status 200
                }
            }
        });
        //chrome.storage.local.remove('payload'); // removes the given key value
    });
})();
