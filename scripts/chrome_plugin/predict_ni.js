// ContentScript
(function() {

    let selected_Text;
    selected_Text = window.getSelection().toString();
    chrome.storage.local.set({text_payload: selected_Text});
    
    const req = new XMLHttpRequest();
    const baseUrl = "http://localhost:5000/predict_nat_id"; // Add rest endpoint here
    
    var data = {sentence: selected_Text, modelNumber: "0"};
    req.open(method="POST", url=baseUrl, true);
    req.setRequestHeader("Content-Type", "application/json");
    req.send(JSON.stringify(data));
    req.onreadystatechange = function() {
        if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {         
            chrome.storage.local.set({response_payload_ni: this.response});
        }
    }
})();
