import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

import getData from './get-data';

async function run() {
  const data = await getData();

  console.log(data);

  const values = data.map(d => {
    return {
      x: d.horsepower,
      y: d.LPer100km,
    };
  });

  //   tfvis.render.scatterplot(
  //     { name: 'Horsepower vs L/100km' },
  //     { values },
  //     {
  //       xLabel: 'Horsepower',
  //       yLabel: 'L/100km',
  //       height: 300,
  //     }
  //   );

  const model = createModel();
  tfvis.show.modelSummary({ name: 'Model summary' }, model);

  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  // Train the model
  await trainModel(model, inputs, labels);
  console.log('Done Training');

  // Make some predictions using the model and compare them to the
  // original data
  testModel(model, data, tensorData);
}

document.addEventListener('DOMContentLoaded', run);

async function trainModel(model, inputs, labels) {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 32;
  const epochs = 100;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    // callbacks: tfvis.show.fitCallbacks({ name: 'Training performance' }, ['loss'], {
    //   height: 200,
    //   callbacks: ['onEpochEnd'],
    // }),
  });
}

function convertToTensor(data) {
  return tf.tidy(() => {
    // Shuffle our data. This will help each batch to have a good variation which will give better training
    tf.util.shuffle(data);

    // Convert our data to tensors
    const inputs = data.map(d => d.horsepower);
    const labels = data.map(d => d.LPer100km);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    // Normalize the data
    // i.e -> [0,1]
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    // This can be done before creating the tensors. But tensors have theses nice functions which will help us!
    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}

// Now we want to train a model which will predict L/100km given the horsepower
function createModel() {
  const model = tf.sequential();

  // Single input layer
  // A dense layer will multiply our input with a matrix (weights). Tänk lagren från DL-kursen
  // inputShape är antalet variabler i input
  // units - hur stor våran weight matrix skall vara. Nu är den lika stor som antal input, altså en vikt för varje variabel
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

  // Output layer
  // Units sätts till 1 för att vi vill ha en variabel ut
  model.add(tf.layers.dense({ units: 1, useBias: true }));

  return model;
}

function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100, 1]));

    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);

    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const originalPoints = inputData.map(d => ({
    x: d.horsepower,
    y: d.LPer100km,
  }));

  tfvis.render.scatterplot(
    { name: 'Model Predictions vs Original Data' },
    { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
    {
      xLabel: 'Horsepower',
      yLabel: 'LPer100km',
      height: 300,
    }
  );
}
