// Functions and stuff comes here
function getSelectionText() {
    var text = "";
    if (window.getSelection) {
        text = window.getSelection().toString();
    } else if (document.selection && document.selection.type != "Control") {
        text = document.selection.createRange().text;
    }
    return text;
}

document.addEventListener('DOMContentLoaded', function() {
    var checkPageButton = document.getElementById('Predict');
    checkPageButton.addEventListener('click', function() {
        // Everything that should happen on button click comes here
        let text;
        //text = window.getSelection().toString();
        text = chrome.extension.getBackgroundPage().window.getSelection().toString()
        alert(text);
        alert("test")

    }, false);
}, false);
