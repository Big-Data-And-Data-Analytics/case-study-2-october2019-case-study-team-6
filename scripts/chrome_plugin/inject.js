(function() {

    let selected_Text;
    selected_Text = window.getSelection().toString();

    const req = new XMLHttpRequest();
    const baseUrl = "http://localhost:8000?CUTHERE " + selected_Text + " CUTHERE";

    req.open(method="POST", url=baseUrl, true);
    req.setRequestHeader("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8");
    req.send(selected_Text);
    req.onreadystatechange = function() { // Call a function when the state changes.
        if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
            alert("Got response 200!");
        }
    }
})();
