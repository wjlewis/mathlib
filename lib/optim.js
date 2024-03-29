export class Sgd {
  #parameters;
  #lr;

  constructor(parameters, lr) {
    this.#parameters = parameters;
    this.#lr = lr;
  }

  zeroGrad() {
    for (const param of this.#parameters) {
      param.grad = 0;
    }
  }

  step() {
    for (const param of this.#parameters) {
      param.value -= this.#lr * param.grad;
    }
  }
}
