(function() {

    let selected_Text;
    selected_Text = window.getSelection().toString();
    alert(selected_Text);
    
    const createCsvWriter = require('csv-writer').createObjectCsvWriter;
    const csvWriter = createCsvWriter({
    path: 'out.csv',
    header: [
        {id: 'text', title: 'Text'}
    ],
    csvWriter.writeRecords(data).then(()=> console.log('The CSV file was written successfully'));
});

})();
