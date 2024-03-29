import D from './autodiff.js';
import { zip } from './tools.js';

export class CrossEntropyLoss {
  apply(output, label) {
    const s = D.sum(zip(output, label, (out, l) => l.times(out.log())));
    return s.neg();
  }
}

export const quux = 42;
