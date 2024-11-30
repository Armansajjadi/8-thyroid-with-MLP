let weights = [];

let inputNeuron = 0

let accuracyData = [];
let epochsData = [];
let chart;

let currentAccuracy=0


let middleNeuron = 20

let outputNeuron = 0

let alfa = 0.2;

let v = []

let dataForTrain = []

let b = []

let biasHidden = []

const doneTrainBtn = document.getElementById("doneTrainBtn")

window.addEventListener("load", () => {

    fetch('trainData.json')
        .then(response => {
            if (!response.ok) {
                throw new Error('File not found');
            }
            return response.json();
        })
        .then(data => {
            console.log("Weights:", data.weights);
            console.log("V:", data.v);
            console.log("b:", data.b);
            console.log("biasHidden:", data.biasHidden);

            weights = data.weights;
            v = data.v;
            b = data.b;
            biasHidden = data.biasHidden
            testSets = data.testSets
            inputNeuron = data.inputNeuron
            outputNeuron = data.outputNeuron
            epochsData = data.epochsData
            accuracyData = data.accuracyData
            initializeChart()
            test();
            doneTrainBtn.classList.add("hidden")
        })
        .catch(error => {
            fetch("DataSets.json")
                .then(res => {
                    return res.json()
                }).then(items => {
                    testSets = [...getRandomSubarray(items, 1080)]
                    validationSets = [...getRandomSubarray(items, 1080)]
                    trainingSets = [...items]
                    inputNeuron = trainingSets[0].data.length
                    outputNeuron = trainingSets[0].y.length
                })
                .catch(er => {
                    console.log(er);
                })
        });
});

function getRandomSubarray(originalArray, size) {
    let subarray = [];
    for (let i = 0; i < size; i++) {
        let randomIndex = Math.floor(Math.random() * originalArray.length);
        subarray.push(originalArray[randomIndex]);
        originalArray.splice(randomIndex, 1);
    }
    return subarray;
}


function test() {
    let counter = 0;


    let x = Array(inputNeuron).fill(0),
        z = Array(middleNeuron).fill(0),
        y = Array(outputNeuron).fill(0);
    let zNetinput = 0

    testSets.forEach(item => {
        item.data.forEach((num, index) => {
            x[index] = num
        })

        zNetinput = 0

        for (let j = 0; j < middleNeuron; j++) {
            zNetinput = b[j];
            for (let i = 0; i < inputNeuron; i++) {
                zNetinput += (x[i] * weights[i][j]);
            }
            z[j] = (bipolarSigmoid(zNetinput));
        }

        let yNetinput = 0

        for (let k = 0; k < outputNeuron; k++) {
            yNetinput = biasHidden[k]
            for (let j = 0; j < middleNeuron; j++) {
                yNetinput += (z[j] * v[j][k])
            }
            y[k] = bipolarSigmoid(yNetinput);
            if (y[k] > 0.5) {
                y[k] = 1
            } else {
                y[k] = 0
            }
        }

        if (isArraysEqual(item.y, y)) {
            counter++
        }

    })

    const accuracyValue = document.getElementById("accuracyValue");
    accuracyValue.innerHTML = `${((counter / testSets.length) * 100).toFixed(2)}%`;
    currentAccuracy=(((counter / testSets.length) * 100).toFixed(2))
}

function isArraysEqual(arr1, arr2) {
    for (let i = 0; i < arr1.length; i++) {
        if (arr1[i] !== arr2[i]) return false;
    }
    return true;
}



doneTrainBtn.addEventListener("click", () => {

    let epoch = 0

    let training = true

    let deltaW = [];

    initializeChart();

    for (let i = 0; i < inputNeuron; i++) {
        let temparr = []
        for (let j = 0; j < middleNeuron; j++) {
            temparr[j] = (Math.random() - 0.5)
        }
        weights[i] = [...temparr];
        deltaW.push(temparr);
    }
    let deltaBiasHidden = [];

    for (let k = 0; k < outputNeuron; k++) {
        deltaBiasHidden.push(0);
        biasHidden.push(Math.random() - 0.5)
    }

    let deltaBias = []
    for (let j = 0; j < middleNeuron; j++) {
        b.push(Math.random() - 0.5)
        deltaBias.push(0)
    }

    for (let j = 0; j < middleNeuron; j++) {
        let temp = [];
        for (k = 0; k < outputNeuron; k++) {
            temp.push(Math.random() - 0.5);
        }
        v[j] = [...temp];
    }
    let deltaV = []; 
    for (let j = 0; j < middleNeuron; j++) {
        deltaV[j] = [];  
        for (let k = 0; k < outputNeuron; k++) {
            deltaV[j][k] = 0; 
        }
    }


    let x = Array(inputNeuron).fill(0),
        z = Array(middleNeuron).fill(0),
        y = Array(outputNeuron).fill(0),
        javab = Array(outputNeuron).fill(0),
        deltaKuchak = Array(outputNeuron).fill(0),
        D = Array(middleNeuron).fill(0),
        zNetInputs = Array(middleNeuron).fill(0);
    let zNetinput = 0
    let yNetinput = 0

    let currentBatch = 0;

    while (training) {

        trainingSets.forEach(item => {


            /****************************forwarding************************************************* */

            item.data.forEach((num, index) => {
                x[index] = num
            })
            // console.log(x);
            zNetinput = 0

            for (let j = 0; j < middleNeuron; j++) {
                zNetinput = b[j];
                for (let i = 0; i < inputNeuron; i++) {
                    zNetinput += (x[i] * weights[i][j]);
                }
                z[j] = (bipolarSigmoid(zNetinput));
                zNetInputs[j] = zNetinput;
            }

            yNetinput = 0

            for (let k = 0; k < outputNeuron; k++) {
                yNetinput = biasHidden[k]
                for (let j = 0; j < middleNeuron; j++) {
                    yNetinput += (z[j] * v[j][k])
                }
                y[k] = bipolarSigmoid(yNetinput);


                /*/---------------------------------back propagation------------------------------------------------------------------------------------*/


                let moshtag = moshtagbipolarSigmoid(yNetinput)
                deltaKuchak[k] = ((item.y[k] - y[k]) * moshtag);
                for (let j = 0; j < middleNeuron; j++) {
                    deltaV[j][k] = (alfa * deltaKuchak[k] * z[j])
                }
                deltaBiasHidden[k] = alfa * deltaKuchak[k]
            }

            let deltaKuchakHidden = Array(middleNeuron).fill(0);
            for (let j = 0; j < middleNeuron; j++) {
                deltaKuchakHidden[j] = 0;
                for (let k = 0; k < outputNeuron; k++) {
                    D[j] = deltaKuchak[k] * v[j][k]; // Contribution from output neuron k
                    let temp = moshtagbipolarSigmoid(zNetInputs[j]); // Derivative of activation function
                    deltaKuchakHidden[j] += D[j] * temp; // Accumulate contribution
                }
            }


            for (let i = 0; i < inputNeuron; i++) {
                for (let j = 0; j < middleNeuron; j++) {
                    deltaW[i][j] = alfa * deltaKuchakHidden[j] * x[i];
                    deltaBias[i] = alfa * deltaKuchakHidden[j];
                }
            }


            /**********************************Updating********************************************************************/


            for (let i = 0; i < inputNeuron; i++) {
                for (let j = 0; j < middleNeuron; j++) {
                    weights[i][j] += deltaW[i][j];
                }
            }

            for (let j = 0; j < middleNeuron; j++) {
                b[j] += deltaBias[j]
            }


            for (let j = 0; j < middleNeuron; j++) {
                for (let k = 0; k < outputNeuron; k++) {
                    v[j][k] += deltaV[j][k];
                }
            }

            for (let k = 0; k < outputNeuron; k++) {
                biasHidden[k] += deltaBiasHidden[k]
            }

        })



        epoch++;

        if(epoch == 1){
            test();
            updateChart(epoch, currentAccuracy);
        }


        console.log(epoch);
        if (epoch % 10 == 0) {
            test();
            updateChart(epoch, currentAccuracy);
            let batchSize = 50;

            let startIndex = currentBatch * batchSize;
            let endIndex = Math.min(startIndex + batchSize, validationSets.length);

            let allCorrect = true;

            for (let i = startIndex; i < endIndex; i++) {
                let item = validationSets[i];
                item.data.forEach((num, index) => {
                    x[index] = num;
                });

                for (let j = 0; j < middleNeuron; j++) {
                    zNetinput = b[j];
                    for (let i = 0; i < inputNeuron; i++) {
                        zNetinput += x[i] * weights[i][j];
                    }
                    z[j] = bipolarSigmoid(zNetinput);
                }

                for (let k = 0; k < outputNeuron; k++) {
                    yNetinput = biasHidden[k];
                    for (let j = 0; j < middleNeuron; j++) {
                        yNetinput += z[j] * v[j][k];
                    }
                    y[k] = bipolarSigmoid(yNetinput);

                    javab[k] = y[k] > 0.5 ? 1 : 0;
                }


                if (!isArraysEqual(item.y, javab)) {
                    allCorrect = false;
                    break;
                }
            }

            if (allCorrect) {
                training = false;
            } else {
                currentBatch++;
                if (currentBatch * batchSize >= validationSets.length) {
                    currentBatch = 0;
                }
            }

        }
        if (epoch == 5000) {
            break
        }
    }

    console.log("finished at", epoch, "epoches");

    const data = {
        weights,
        b,
        biasHidden,
        v,
        testSets,
        inputNeuron,
        outputNeuron,
        epochsData,
        accuracyData
    };

    const jsonData = JSON.stringify(data);

    const blob = new Blob([jsonData], { type: 'application/json' });

    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'trainData.json';
    document.body.appendChild(a);
    a.click();

    document.body.removeChild(a);

    URL.revokeObjectURL(url);

    console.log("Data has been saved as JSON file!");

    doneTrainBtn.classList.add("hidden")
    test()

});

function bipolarSigmoid(x) {
    return (2 / (1 + Math.exp(-x))) - 1;
}

function moshtagbipolarSigmoid(x) {
    const sigmoidValue = bipolarSigmoid(x);
    return ((1 + sigmoidValue) * (1 - sigmoidValue)) / 2;
}

function initializeChart() {
    const ch=document.getElementById('accuracyChart')
    const ctx = ch.getContext('2d');
    ch.classList.remove("hidden");
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochsData,
            datasets: [{
                label: 'Accuracy',
                data: accuracyData,
                borderColor: 'blue',
                backgroundColor: 'rgba(0, 0, 255, 0.1)',
                fill: true,
                pointRadius: 3,
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'Epoch' } },
                y: { title: { display: true, text: 'Accuracy (%)' }, beginAtZero: true }
            }
        }
    });
}

function updateChart(epoch, accuracy) {
    epochsData.push(epoch);
    accuracyData.push(accuracy);
    chart.update();
}
