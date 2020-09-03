var loadFile = function (event) {
    $("#result").text("")
    var output = document.getElementById('output');
    output.src = URL.createObjectURL(event.target.files[0]);
    output.onload = function () {
        URL.revokeObjectURL(output.src)
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
    model = await tf.loadLayersModel(
        "https://bbuluttekin.github.io/MSc-Project/static/model.json");
    $('#spinner').parent().hide();
})();

var predict = async function () {
    let image = $("#output").get(0);
    let tensor = tf.browser.fromPixels(image);
    let resizedTensor = tf.image.resizeBilinear(tensor, [224, 224])
        .mul(1 / 255)
        .expandDims();
    let prediction = await model.predict(resizedTensor).data();
    if (prediction <= 0.5) {
        // Normal
        $("#result").text("Result: No Pneumonia detected.")
    } else {
        // Pneumonia 
        $("#result").text("Result: Pneumonia detected.")
    }
};