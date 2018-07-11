
const epochsPerIteration = 10;
const batchSize = 100;

// Generate some random colors for visualization of predictions
let randomColors = [];
with (Math) for (let i = 0; i < 60; i++) {
    randomColors.push({ r: random(), g: random(), b: random() });
}

// Get color definition training data from github
fetch('https://raw.githubusercontent.com/fmaul/tf-color-classification/master/colors.json').then(result => {
    return result.json();
}).then(data => {
/*
    randomColors.map(target => {
        let distances = data.map(c => {
            let d = Math.sqrt(Math.pow(target.r*255 - c.r, 2)+Math.pow(target.g*255 - c.g, 2)+Math.pow(target.b*255 - c.b, 2));
            return { d, color: c.color }
        }).sort((a,b) => a.d < b.d ? -1 : 1);

        //console.log(distances);
        target.bestMatch = distances[0].color;
    });
        */
    setup(data);
});

function setup(data) {
    let labels = buildIndexOfAllLabels(data);
    let model = buildAndCompileClassificationModel(labels);

    let xs = tf.tidy(() => tf.tensor2d(data.map(c => [c.r / 255, c.g / 255, c.b / 255])));
    let ys = tf.tidy(() => tf.oneHot(data.map(c => c.labelIdx), labels.length).toFloat());

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
    model.summary();
    return model;
}

function trainLoop(labels, model, xs, ys, iteration) {
    if (iteration > 100) return;

    model.fit(xs, ys, {
        shuffle: true,
        epochs: epochsPerIteration,
        batchSize
    }).then(training => {
        $("#header").text("Epoch " + iteration * epochsPerIteration + " Loss: " + training.history.loss[0]);
        printPredictions(labels, model);
        window.setTimeout(() => trainLoop(labels, model, xs, ys, iteration + 1), 10);
    });
}

function printPredictions(labels, model) {
    $("#main").empty();
    tf.tidy(() => {
        let prediction = model.predict(tf.tensor2d(randomColors.map(c => [c.r, c.g, c.b])));

        randomColors.forEach((td, i) => {
            let probabilities = tf.slice(prediction, i, 1).dataSync(); // TypedArray
            let info = buildMaxPredictionInfo(probabilities, labels);
            let color = "rgb(" + Math.floor(td.r * 255) + "," + Math.floor(td.g * 255) + "," + Math.floor(td.b * 255) + ")";
            $("#main").append("<div style=\"background:" + color + "\">" + info + "</div>");
        });
    });
}

/* argMax with max 3 results, threshold 0.1 and formatting */
function buildMaxPredictionInfo(probabilities, labels) {
    return Array.from(probabilities, (p, i) => ({ p, i }))
        .sort((a, b) => a.p < b.p)
        .slice(0, 3)
        .filter(e => e.p > 0.1)
        .map(e => labels[e.i] + ": " + round3(e.p))
        .map((s, i) => (i == 0) ? "<b>" + s + "</b>" : s)
        .join(", ");
}

function round3(v) {
    return Math.round(v * 1000) / 1000;
}
