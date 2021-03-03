(function() {

    let selected_Text;
    selected_Text = window.getSelection().toString();
    alert(selected_Text);
    
    const req = new XMLHttpRequest();
//    const baseUrl = "http://localhost:5000/models_IM";
    const baseUrl = "http://localhost:5000/predict_id_motive";
    var body = XMLHttpRequest.response;

//    var param = "sentence=" + selected_Text + "&modelNumber='0'"
    var data = {sentence: selected_Text, modelNumber: "0"};
//    var data = "{'sentence': 'selected_Text', 'modelNumber': '0'}";
//    http://localhost:5000/predict_id_motive?sentence="selected_Text"&modelNumber="0"

    req.open(method="POST", url=baseUrl, true);
//    req.open(method="GET", url=baseUrl, true);
//    req.setRequestHeader("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8");
    req.setRequestHeader("Content-Type", "application/json");
//    alert(JSON.stringify(data))
    req.send(JSON.stringify(data));
//    req.send(data);
    req.onreadystatechange = function() { // Call a function when the state changes.
        if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
            alert("Got response 200!");
            alert(this.responseText)
        }
    }
})();
