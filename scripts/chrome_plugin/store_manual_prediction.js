// ContentScript
(function() {

    chrome.storage.local.get('class_payload', function (items) {
        // console.log(items.class_payload)
        const my_identity_motive = items.class_payload
        chrome.storage.local.remove('class_payload');
        
        chrome.storage.local.get('text_payload', function (items) {
            // console.log(items.text_payload);
            const my_text = items.text_payload;
            chrome.storage.local.remove('text_payload');
            
            chrome.storage.local.get('country_payload', function (items) {
                // console.log(items.country_payload);
                const my_country = items.country_payload;
                chrome.storage.local.remove('country_payload');

                const req = new XMLHttpRequest();
                const baseUrl = "http://localhost:5000/save_prediction"; // Endpoint for storing to DB
            
                var data = {sentence: my_text, identity_motive: my_identity_motive, national_identity: my_country};
                req.open(method="POST", url=baseUrl, true);
                req.setRequestHeader("Content-Type", "application/json");
                req.send(JSON.stringify(data));
                req.onreadystatechange = function() {
                    if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
                        // Also something could be here for status 200
                        alert(this.responseText)
                    }
                }
            });
        });
    });
})();
