import D from './autodiff.js';

describe('autodiff', () => {
  const cases = [
    {
      name: 'x => x',
      fn: x => x,
      arg: D.scalar(42),
      expected: 1,
    },
    {
      name: 'x => x^2',
      fn: x => x.pow(2),
      arg: D.scalar(2),
      expected: 4,
    },
    {
      name: 'x => 2x^2 - 3x + 4',
      fn: x =>
        D.scalar(2)
          .times(x.pow(2))
          .minus(D.scalar(3).times(x))
          .plus(D.scalar(4)),
      arg: D.scalar(2),
      expected: 5,
    },
    {
      name: 'x => e^(2x)',
      fn: x => D.scalar(2).times(x).exp(),
      arg: D.scalar(1),
      expected: 2 * Math.exp(2),
    },
  ];

  test.each(cases)('Differentiates $name', ({ fn, arg, expected }) => {
    const y = fn(arg);
    y.backward();
    return expect(arg.grad).toBe(expected);
  });
});
