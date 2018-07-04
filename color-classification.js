// Generate some random colors for visualization of predictions
let validationData = [];
for (let i = 0; i < 100; i++) {
    validationData.push({ r: Math.random(), g: Math.random(), b: Math.random() });
}

// Get color definition training data from github
fetch('https://raw.githubusercontent.com/fmaul/tf-color-classification/master/colors.json').then(result => {
    return result.json();
}).then(data => {
    setup(data);
});

function setup(data) {
    let labels = buildIndexOfAllLabels(data);
    let model = buildAndCompileClassificationModel(labels);

    let xs = tf.tidy(() => tf.tensor2d(data.map(c => [c.r / 255, c.g / 255, c.b / 255])));
    let ys = tf.tidy(() => tf.oneHot(tf.tensor1d(data.map(c => c.labelIdx), 'int32'), labels.length).toFloat());
    
    trainLoop(labels, model, xs, ys, 0);
}

function buildIndexOfAllLabels(data) {
    let labels = [];
    data.forEach(d => {
        let i = labels.findIndex((e => e == d.color));
        if (i >= 0) d.labelIdx = i;
        else {
            d.labelIdx = labels.length;
            labels.push(d.color);
        }
    });
    console.log(labels);
    return labels;
}

function buildAndCompileClassificationModel(labels) {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [3],
        units: labels.length * 2,
        activation: 'relu'
    }));
    model.add(tf.layers.dense({
        units: labels.length,
        activation: 'softmax'
    }));
    model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError' // tf.losses.softmaxCrossEntropy
    });
    return model;
}

function trainLoop(labels, model, xs, ys, iteration) {
    if (iteration > 100) return;

    model.fit(xs, ys, {
        shuffle: true,
        epochs: 10,
        batchSize: 100
    }).then(training => {
        $("#header").text("Iteration " + iteration + " Loss: " + training.history.loss[0]);
        printPredictions(labels, model);
        window.setTimeout(() => trainLoop(labels, model, xs, ys, iteration + 1), 10);
    });
}

function printPredictions(labels, model) {
    $("#main").empty();
    tf.tidy(() => {
        let prediction = model.predict(tf.tensor2d(validationData.map(td => [td.r, td.g, td.b])));

        validationData.forEach((td, i) => {
            let probabilities = tf.slice(prediction, i, 1).dataSync(); // TypedArray
            let info = buildMaxPredictionInfo(probabilities, labels);
            let color = "rgb(" + Math.floor(td.r * 255) + "," + Math.floor(td.g * 255) + "," + Math.floor(td.b * 255) + ")";
            $("#main").append("<div style=\"background:" + color + "\">" + info + "</div>");
        });
    });
}
/* argMax with max 3 results, threshold 0.1 and formating */
function buildMaxPredictionInfo(probabilities, labels) {
    return Array.from(probabilities, (p, i) => ({ p, i }))
        .sort((a, b) => a.p < b.p)
        .slice(0, 3)
        .filter(e => e.p > 0.1)
        .map(e => "" + labels[e.i] + ": " + round3(e.p))
        .map((s, i) => (i == 0) ? "<b>" + s + "</b>" : s)
        .join(", ");
}

function round3(v) {
    return Math.round(v * 1000) / 1000;
}
