(function() {

    let selected_Text;
    selected_Text = window.getSelection().toString();
    alert(selected_Text);
    
    const req = new XMLHttpRequest();

    const baseUrl = "http://localhost:5000/ENDPOINT"; // Add rest endpoint here
    
    var data = {sentence: selected_Text, modelNumber: "0"};
    req.open(method="POST", url=baseUrl, true);
    req.setRequestHeader("Content-Type", "application/json");
    req.send(JSON.stringify(data));
    req.onreadystatechange = function() { // Call a function when the state changes.
        if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
            alert("Got response 200!");
            alert(this.responseText)
        }
    }
})();
