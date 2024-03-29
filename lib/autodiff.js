/**
 * Construct expressions that can be automatically differentiated.
 */
export default class D {
  #deps;
  #grad;
  #backward;

  constructor(value, deps = []) {
    this.value = value;
    this.grad = 0;
    this.#deps = deps;
    this.#backward = () => {};
  }

  static scalar(value) {
    return new D(value);
  }

  backward() {
    const sorted = [];
    const seen = new Set();

    const visit = node => {
      if (!seen.has(node)) {
        seen.add(node);
        for (const dep of node.#deps) {
          visit(dep);
        }
        sorted.push(node);
      }
    };
    visit(this);
    sorted.reverse();

    this.grad = 1;
    for (const node of sorted) {
      node.#backward();
    }
  }

  static sum(xs) {
    return xs.reduce((sum, x) => sum.plus(x), D.scalar(0));
  }

  plus(rhs) {
    const d = new D(this.value + rhs.value, [this, rhs]);
    d.#backward = () => {
      this.grad += d.grad;
      rhs.grad += this.grad;
    };
    return d;
  }

  times(rhs) {
    const d = new D(this.value * rhs.value, [this, rhs]);
    d.#backward = () => {
      this.grad += rhs.value * d.grad;
      rhs.grad += this.value * d.grad;
    };
    return d;
  }

  pow(n) {
    const d = new D(this.value ** n, [this]);
    d.#backward = () => {
      this.grad += n * this.value ** (n - 1) * d.grad;
    };
    return d;
  }

  minus(rhs) {
    return this.plus(rhs.neg());
  }

  neg() {
    const d = new D(-this.value, [this]);
    d.#backward = () => {
      this.grad += -d.grad;
    };
    return d;
  }

  inv() {
    const d = new D(1 / this.value, [this]);
    d.#backward = () => {
      this.grad += (-1 / this.value ** 2) * d.grad;
    };
    return d;
  }

  div(rhs) {
    return this.times(rhs.inv());
  }

  exp() {
    const e = Math.exp(this.value);
    const d = new D(e, [this]);
    d.#backward = () => {
      this.grad += e * d.grad;
    };
    return d;
  }

  log() {
    const d = new D(Math.log(this.value), [this]);
    d.#backward = () => {
      this.grad += (1 / this.value) * d.grad;
    };
    return d;
  }
}
