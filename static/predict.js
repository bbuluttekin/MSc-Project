var loadFile = function (event) {
    var output = document.getElementById('output');
    output.src = URL.createObjectURL(event.target.files[0]);
    output.onload = function () {
        URL.revokeObjectURL(output.src) // free memory
    }


    // $('#spinner').parent().hide()
};

$("#output").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;

        // $("#selected-image").attr("src", dataURL);
        // $("#prediction-list").empty();
    }
    let file = $("#inputGroupFile01").prop('files')[0];
    reader.readAsDataURL(file);
});


let model;
(async function () {
    model = await tf.loadLayersModel("");
    $('#spinner').parent().hide();
})();

var predict = async function () {
    let image = $("#output").get(0);
    let tensor = tf.browser.fromPixels(image);
};