import D from './autodiff.js';
import { zip, range, uniform } from './tools.js';

export class Module {
  get parameters() {
    throw new Error('unimplemented');
  }

  zeroGrad() {
    for (const param of this.parameters) {
      param.grad = 0;
    }
  }
}

export class Neuron extends Module {
  #weights;
  #bias;

  constructor(inCount) {
    super();
    this.#weights = range(0, inCount).map(() => D.scalar(uniform(-1, 1)));
    this.#bias = D.scalar(uniform(-1, 1));
  }

  apply(xs) {
    return D.sum(zip(this.#weights, xs, (w, x) => w.times(x))).plus(this.#bias);
  }

  get parameters() {
    return [...this.#weights, this.#bias];
  }
}

export class Linear extends Module {
  #neurons;

  constructor(inCount, outCount) {
    super();
    this.#neurons = range(0, outCount).map(() => new Neuron(inCount));
  }

  apply(xs) {
    return this.#neurons.map(n => n.apply(xs));
  }

  get parameters() {
    return this.#neurons.flatMap(n => n.parameters);
  }
}

export class Softmax extends Module {
  apply(xs) {
    const denom = D.sum(xs.map(x => x.exp()));
    return xs.map(x => x.exp().div(denom));
  }

  get parameters() {
    return [];
  }
}
