import D from './autodiff.js';
import { Linear, Softmax, Module } from './net.js';
import { CrossEntropyLoss } from './loss.js';
import { Sgd } from './optim.js';

class LogReg extends Module {
  constructor() {
    super();
    this.lin = new Linear(2, 3);
    this.soft = new Softmax();
  }

  apply(xs) {
    const ys = this.lin.apply(xs);
    return this.soft.apply(ys);
  }

  get parameters() {
    return [...this.lin.parameters, ...this.soft.parameters];
  }
}

//

const data = [
  { point: [1.2, 0.5], label: [1, 0, 0] },
  { point: [2.4, 1.9], label: [0, 1, 0] },
  { point: [0.8, 0.6], label: [1, 0, 0] },
  { point: [3, 2.5], label: [0, 0, 1] },
  { point: [1.5, 0.9], label: [1, 0, 0] },
  { point: [2.8, 2], label: [0, 1, 0] },
  { point: [0.5, 0.3], label: [1, 0, 0] },
  { point: [2.7, 2.3], label: [0, 1, 0] },
  { point: [1, 0.8], label: [1, 0, 0] },
  { point: [2.9, 2.1], label: [0, 1, 0] },
  { point: [1.4, 0.7], label: [1, 0, 0] },
  { point: [2.6, 1.8], label: [0, 1, 0] },
  { point: [0.9, 0.4], label: [1, 0, 0] },
  { point: [2.5, 2.2], label: [0, 1, 0] },
  { point: [1.1, 0.9], label: [1, 0, 0] },
  { point: [3.1, 2.6], label: [0, 0, 1] },
  { point: [1.3, 0.6], label: [1, 0, 0] },
  { point: [2.7, 1.7], label: [0, 1, 0] },
  { point: [0.7, 0.2], label: [1, 0, 0] },
  { point: [2.3, 2.4], label: [0, 0, 1] },
  { point: [1.6, 0.8], label: [1, 0, 0] },
  { point: [2.8, 1.6], label: [0, 1, 0] },
  { point: [0.6, 0.5], label: [1, 0, 0] },
  { point: [2.2, 2.6], label: [0, 0, 1] },
  { point: [1.7, 0.7], label: [1, 0, 0] },
  { point: [2.7, 1.5], label: [0, 1, 0] },
  { point: [0.4, 0.1], label: [1, 0, 0] },
  { point: [2.1, 2.8], label: [0, 0, 1] },
  { point: [1.8, 0.6], label: [1, 0, 0] },
  { point: [2.6, 1.4], label: [0, 1, 0] },
  { point: [0.3, 0.2], label: [1, 0, 0] },
  { point: [2, 2.7], label: [0, 0, 1] },
  { point: [1.9, 0.5], label: [1, 0, 0] },
  { point: [2.5, 1.3], label: [0, 1, 0] },
  { point: [0.2, 0.3], label: [1, 0, 0] },
  { point: [1.8, 2.9], label: [0, 0, 1] },
  { point: [2.4, 1.2], label: [0, 1, 0] },
  { point: [0.1, 0.4], label: [1, 0, 0] },
  { point: [1.7, 3], label: [0, 0, 1] },
  { point: [2.3, 1.1], label: [0, 1, 0] },
  { point: [0, 0.5], label: [1, 0, 0] },
  { point: [1.6, 3.1], label: [0, 0, 1] },
  { point: [2.2, 1], label: [0, 1, 0] },
  { point: [-0.1, 0.6], label: [1, 0, 0] },
  { point: [1.5, 3.2], label: [0, 0, 1] },
  { point: [2.1, 0.9], label: [0, 1, 0] },
  { point: [-0.2, 0.7], label: [1, 0, 0] },
  { point: [1.4, 3.3], label: [0, 0, 1] },
  { point: [2, 0.8], label: [0, 1, 0] },
  { point: [-0.3, 0.8], label: [1, 0, 0] },
  { point: [1.3, 3.4], label: [0, 0, 1] },
  { point: [1.9, 0.7], label: [0, 1, 0] },
  { point: [-0.4, 0.9], label: [1, 0, 0] },
  { point: [1.2, 3.5], label: [0, 0, 1] },
  { point: [1.8, 0.6], label: [0, 1, 0] },
  { point: [-0.5, 1], label: [1, 0, 0] },
  { point: [1.1, 3.6], label: [0, 0, 1] },
  { point: [1.7, 0.5], label: [0, 1, 0] },
  { point: [-0.6, 1.1], label: [1, 0, 0] },
  { point: [1, 3.7], label: [0, 0, 1] },
  { point: [1.6, 0.4], label: [0, 1, 0] },
  { point: [-0.7, 1.2], label: [1, 0, 0] },
  { point: [0.9, 3.8], label: [0, 0, 1] },
  { point: [1.5, 0.3], label: [0, 1, 0] },
  { point: [-0.8, 1.3], label: [1, 0, 0] },
  { point: [0.8, 3.9], label: [0, 0, 1] },
  { point: [1.4, 0.2], label: [0, 1, 0] },
  { point: [-0.9, 1.4], label: [1, 0, 0] },
];

const valData = [
  { point: [0.7, 4], label: [0, 0, 1] },
  { point: [1.3, 0.1], label: [0, 1, 0] },
  { point: [-1, 1.5], label: [1, 0, 0] },
  { point: [0.6, 4.1], label: [0, 0, 1] },
  { point: [1.2, 0], label: [0, 1, 0] },
  { point: [-1.1, 1.6], label: [1, 0, 0] },
  { point: [0.5, 4.2], label: [0, 0, 1] },
  { point: [1.1, -0.1], label: [0, 1, 0] },
  { point: [-1.2, 1.7], label: [1, 0, 0] },
  { point: [0.4, 4.3], label: [0, 0, 1] },
  { point: [1, -0.2], label: [0, 1, 0] },
  { point: [-1.3, 1.8], label: [1, 0, 0] },
  { point: [0.3, 4.4], label: [0, 0, 1] },
  { point: [0.9, -0.3], label: [0, 1, 0] },
  { point: [-1.4, 1.9], label: [1, 0, 0] },
  { point: [0.2, 4.5], label: [0, 0, 1] },
  { point: [0.8, -0.4], label: [0, 1, 0] },
  { point: [-1.5, 2], label: [1, 0, 0] },
  { point: [0.1, 4.6], label: [0, 0, 1] },
  { point: [0.7, -0.5], label: [0, 1, 0] },
];

//

export const model = new LogReg();
const loss = new CrossEntropyLoss();
const optim = new Sgd(model.parameters, 0.1);

function getBatch(size = 10) {
  const batch = new Array(size);
  for (let i = 0; i < size; i++) {
    const idx = Math.floor(Math.random() * data.length);
    const elt = data[idx];
    batch[i] = [elt.point.map(D.scalar), elt.label.map(D.scalar)];
  }
  return batch;
}

export function train() {
  const epochSize = 10;
  for (let i = 0; i < 100; i++) {
    optim.zeroGrad();

    const batch = getBatch(10);
    let l = D.scalar(0);
    for (const [point, label] of batch) {
      const pred = model.apply(point);
      l = l.plus(loss.apply(pred, label));
    }
    l = l.times(D.scalar(1 / batch.length));

    l.backward();

    optim.step();

    if (i % epochSize === 0) {
      console.log(`epoch: ${i / epochSize}; loss: ${l.value}`);
    }
  }
}
