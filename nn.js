'use strict';
/*
  The network: plain arrays and maths, no libraries.
    hidden layers:  a = tanh(w·x + b)
    output:         P = 1 / (1 + e^-z)
    loss:           cross-entropy, trained with backpropagation + gradient descent
*/

const gaussian = () =>
  Math.sqrt(-2 * Math.log(Math.random() + 1e-9)) * Math.cos(2 * Math.PI * Math.random());

class Net {
  constructor(sizes, saved) {
    this.sizes = sizes;
    if (saved) { this.W = saved.W; this.b = saved.b; return; }
    this.W = [];
    this.b = [];
    for (let l = 1; l < sizes.length; l++) {
      const inputs = sizes[l - 1], outputs = sizes[l];
      this.W.push(Array.from({ length: outputs }, () =>
        Array.from({ length: inputs }, () => gaussian() * Math.sqrt(1 / inputs))));
      this.b.push(new Array(outputs).fill(0));
    }
  }

  /** Activations of every layer, input first, output last. */
  forward(x) {
    const layers = [x];
    const last = this.W.length - 1;
    this.W.forEach((matrix, l) => {
      const prev = layers[l];
      layers.push(matrix.map((row, j) => {
        let z = this.b[l][j];
        for (let i = 0; i < prev.length; i++) z += row[i] * prev[i];
        return l === last ? 1 / (1 + Math.exp(-z)) : Math.tanh(z);
      }));
    });
    return layers;
  }

  predict(x) { return this.forward(x)[this.W.length][0]; }

  /** One gradient-descent step on a batch of {x, y} samples. */
  train(batch, lr) {
    const last = this.W.length - 1;
    const gradW = this.W.map(m => m.map(row => row.map(() => 0)));
    const gradB = this.b.map(row => row.map(() => 0));

    for (const sample of batch) {
      const layers = this.forward(sample.x);
      let delta = [layers[last + 1][0] - sample.y];             // output error: P - y
      for (let l = last; l >= 0; l--) {
        delta.forEach((d, j) => {
          gradB[l][j] += d;
          layers[l].forEach((a, i) => { gradW[l][j][i] += d * a; });
        });
        if (l > 0) {                                            // push the error back through tanh
          const upstream = delta;
          delta = layers[l].map((a, i) => {
            let sum = 0;
            upstream.forEach((d, j) => { sum += this.W[l][j][i] * d; });
            return sum * (1 - a * a);
          });
        }
      }
    }

    const scale = lr / batch.length;
    this.W.forEach((matrix, l) => matrix.forEach((row, j) => {
      this.b[l][j] -= scale * gradB[l][j];
      row.forEach((_, i) => { row[i] -= scale * gradW[l][j][i]; });
    }));
  }

  get paramCount() { return this.W.reduce((n, m) => n + m.length * (m[0].length + 1), 0); }

  toJSON() {
    const round = m => m.map(r => (Array.isArray(r) ? round(r) : +r.toFixed(5)));
    return { sizes: this.sizes, W: this.W.map(round), b: this.b.map(round) };
  }
}

module.exports = { Net };
