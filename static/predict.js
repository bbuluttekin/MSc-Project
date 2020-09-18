var loadFile = function (event) {
    $("#result").text("")
    var output = document.getElementById('output');
    output.src = URL.createObjectURL(event.target.files[0]);
    output.onload = function () {
        URL.revokeObjectURL(output.src)
    }
};


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
        $("#result").text(`Result: Pneumonia not detected. (Probability of pneumonia: ${Number(prediction).toFixed(3)})`);
    } else {
        // Pneumonia 
        $("#result").text(`Result: Pneumonia detected. (Probability of pneumonia: ${Number(prediction).toFixed(3)})`);
    }
};