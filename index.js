tf.setBackend('cpu');

const model = tf.sequential();

const hidden1 = tf.layers.dense({ units: 12, inputShape: [4], activation: 'sigmoid' });
const hidden2 = tf.layers.dense({ units: 8, activation: 'sigmoid' });
const output = tf.layers.dense({ units: 3, activation: 'softmax' });

model.add(hidden1);
model.add(hidden2);
model.add(output);

model.compile({ optimizer: tf.train.adam(0.07), loss: 'meanSquaredError' });

let trainDataX;
let trainDataY;

let testDataX;
let testDataY;

let testRatio = 0.15;
let trainEpochs = 32;
let ep = 0;

function loadData(path) {
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
            let data = processData(this.responseText);
            trainDataX = data.train.x;
            trainDataY = data.train.y;
            testDataX = data.test.x;
            testDataY = data.test.y;
            console.log('Data loaded, ' + (trainDataX.length + testDataX.length) + ' records.');
            console.log('Records: train: ' + trainDataX.length + ', test: ' + testDataX.length);
            //train();
        }
    };
    xhttp.open("GET", path, true);
    xhttp.send();
}

function train(e) {
    if (trainDataX && trainDataY)
        model.fit(tf.tensor2d(trainDataX), tf.tensor2d(trainDataY), {
            epochs: e,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log('#' + (ep++).toString().padStart(3, 0) +
                        ' Training loss: ' + logs.loss.toFixed(9))
                    ui.drawLoss(logs.loss)
                }
            }
        });
}

function predict(id) {
    let result = model.predict(tf.tensor2d([testDataX[id]])).dataSync();
    return [result[0], result[1], result[2]];
}

function predictArray(arr) {
    let result = model.predict(tf.tensor2d([arr])).dataSync();
    return [result[0], result[1], result[2]];
}

function arrMaximize(arr) {
    let max = 0;
    for (let i = 0; i < arr.length; i++) {
        if (max < arr[i])
            max = arr[i];
    }
    for (let i = 0; i < arr.length; i++) {
        arr[i] = max == arr[i] ? 1 : 0;
    }
    return arr;
}

function guess(id) {
    return arrMaximize(predict(id));
}

function isEqual(arr1, arr2) {
    if (arr1.length !== arr2.length)
        return false
    for (let i = 0; i < arr1.length; i++) {
        if (arr1[i] !== arr2[i])
            return false;
    }
    return true;
}

function validate() {
    valid = 0;
    for (let i = 0; i < testDataY.length; i++) {
        if (isEqual(guess(i), (testDataY[i])))
            valid++;
    }
    return valid / testDataY.length
}

function processData(data) {
    const lines = data.split(/\r\n|\n/); // cut data into lines
    const headings = lines.shift().split(/\,/); // extract headings
    const test = { x: [], y: [] }, train = { x: [], y: [] };
    for (let line of lines) {
        const values = line.split(/\,/); // separate each line into actual values array
        for (let i = 0; i < values.length; i++) {
            if (!isNaN(values[i])) {
                values[i] = Number.parseFloat(values[i]); // change 'numbers' to actual numbers
            }
        }
        if (Math.random() < testRatio) {
            test.x.push(values.splice(0, 4)); // place first 4 values into x
            test.y.push(values); // place rest 3 into y 
        } else {
            train.x.push(values.splice(0, 4));
            train.y.push(values);
        }

    }
    return { train, test };
}

loadData('data/iris-norm-mc.csv');

const ui = {
    prevLoss: null,
    drawLoss: (loss) => {
        let ctx = cnvLoss.getContext('2d')
        ctx.strokeStyle = '#000000'
        //ctx.beginPath()
        ctx.lineTo(ep * 2, 100 - loss * 200 + .5)
        //ctx.closePath()
        ctx.stroke()
        prevLoss = loss
    }
}

colors = {}
colors.rgbToHex = rgbToHex = function (rgb) {
    var hex = Number(rgb).toString(16);
    if (hex.length < 2) {
        hex = "0" + hex;
    }
    return hex;
};
colors.fullColorHex = function (r, g, b) {
    var red = rgbToHex(r);
    var green = rgbToHex(g);
    var blue = rgbToHex(b);
    return red + green + blue;
};

ui.testSliders = document.querySelectorAll('input.test-slider')
ui.testLabels = document.querySelectorAll('label.test-label')
ui.outPrediction = document.querySelector('input#outPrediction')
ui.onTestSliderInput = function () {
    let inArr = [ui.testSliders[0].valueAsNumber / 100,
    ui.testSliders[1].valueAsNumber / 100,
    ui.testSliders[2].valueAsNumber / 100,
    ui.testSliders[3].valueAsNumber / 100]
    for (let i = 0; i < ui.testLabels.length; i++) {
        ui.testLabels[i].innerText = inArr[i].toFixed(2)
    }
    ui.outPrediction.value = arrMaximize(predictArray(inArr)).toString()
}
ui.onTestSliderChange = function () {
    let inArr = [ui.testSliders[0].valueAsNumber / 100,
    ui.testSliders[1].valueAsNumber / 100,
    ui.testSliders[2].valueAsNumber / 100,
    ui.testSliders[3].valueAsNumber / 100]
    for (let i = 0; i < ui.testLabels.length; i++) {
        ui.testLabels[i].innerText = inArr[i].toFixed(2)
    }
}
for (let slider of ui.testSliders) {
    slider.onchange = ui.onTestSliderInput
    slider.oninput = ui.onTestSliderChange
}

let ctx = cnvLoss.getContext('2d')
ctx.beginPath()
ctx.lineWidth = 1.0
ctx.strokeStyle = '#000000'
ctx.moveTo(0, 100.5)
ctx.lineTo(400, 100.5)
ctx.stroke()
ctx.closePath()


let x = document.createElement('canvas')
x.width = 255;
x.height = 255;
let c = x.getContext('2d')
document.body.appendChild(x)

let size = 15;
function fi() {
    for (let i = 0; i < 256; i += size) {
        for (let j = 0; j < 256; j += size) {
            let p = predictArray([i / 256, j / 256, i/512+.5, 0.35])
            let col = [(p[0] * 256).toFixed(0)]
            col.push((p[1] * 256).toFixed(0))
            col.push((p[2] * 256).toFixed(0))
            c.fillStyle = '#' + colors.fullColorHex(col[0], col[1], col[2])
            c.fillRect(i, j, size, size)
        }
    }
}
fi()