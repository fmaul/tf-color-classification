// Generate some random colors for visualization of predictions
let testData = [];
for (let i = 0; i < 100; i++) {
    testData.push({
        r: Math.random(),
        g: Math.random(),
        b: Math.random()
    });
}

tf.setBackend('webgl');

// Get color definition training data from gist
fetch('https://raw.githubusercontent.com/fmaul/tf-color-classification/master/colors.json').then(result => {
    return result.json();
}).then(data => {
    buildModel(data);
});

function buildModel(data) {
    // create Index of all labels
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

    let xs = tf.tidy(() => tf.tensor2d(data.map(d => [d.r / 255, d.g / 255, d.b / 255])));
    let ys = tf.tidy(() => tf.oneHot(tf.tensor1d(data.map(d => d.labelIdx), 'int32'), labels.length).toFloat());

    console.log("Input: " + xs.shape);
    xs.print();
    console.log("Output: " + ys.shape);
    ys.print();

    const input = tf.input({
        shape: [3]
    });

    const denseLayer1 = tf.layers.dense({
        units: 16,
        activation: 'relu'
    });
    const denseLayer2 = tf.layers.dense({
        units: labels.length,
        activation: 'softmax'
    });
    const output = denseLayer2.apply(denseLayer1.apply(input));
    const model = tf.model({
        inputs: input,
        outputs: output
    });

    // tf.train.sgd(0.5)
    model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError'
    });

    train(labels, model, xs, ys, 0);
}

function train(labels, model, xs, ys, iteration) {
    if (iteration > 80) return;
    // console.log(tf.memory().numTensors);

    const history = model.fit(xs, ys, {
        shuffle: true,
        epochs: 10,
        batchSize: 300
    });

    history.then(h => {
        $("#header").text("Iteration " + iteration + " Loss: " + h.history.loss[0]);
        printPredictions(labels, model);
        window.setTimeout(() => train(labels, model, xs, ys, iteration + 1), 50);
    });
}


function printPredictions(labels, model) {
    $("#main").empty();

    tf.tidy(() => {
        let prediction = model.predict(tf.tensor2d(testData.map(td => [td.r, td.g, td.b]), [testData.length, 3]));
        //let bestMatchIdx = prediction.argMax(1);  argMax works fine but only returns the best result

        testData.forEach((td, i) => {
            // get the prediction for test color i as normal Array
            let probabilities = Array.from(tf.slice(prediction, i, 1).dataSync());
            let info = buildMaxPredictionInfo(probabilities, labels);
            let color = "rgb(" + Math.floor(td.r * 255) + "," + Math.floor(td.g * 255) + "," + Math.floor(td.b * 255) + ")";
            $("#main").append("<div style=\"background:" + color + "\">" + info + "</div>");
        });

    });
}

function buildMaxPredictionInfo(probabilities, labels) {
    return probabilities.map((v, idx) => { return { idx, v } })
        .sort((a, b) => a.v < b.v)
        .slice(0, 3)
        .filter(e => e.v > 0.1)
        .map(e => "" + labels[e.idx] + ": " + round(e.v))
        .map((s,i) => (i==0) ? "<b>"+s+"</b>" : s )
        .join(", ");
}

function round(v) {
    return Math.round(v * 1000) / 1000;
}
