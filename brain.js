'use strict';
/*
  ONE network that learns what your text means.

  Layer 1  word meanings   each word = the average of its spelling pieces (3-5 letter chunks)
                           looked up in a learned table. Trained by predicting the words
                           around it:   P(neighbour | word) = sigmoid(vec(word) · out[neighbour])
                           Words used in similar places get similar numbers (ship ≈ boat),
                           and unseen words (sailing, sailors) still work through their pieces.
  Layer 2  sentence        average of the word meanings, rare words count more.
  Layer 3  judge           small network: sentence meaning -> P(same meaning as your text).
*/
const { Net } = require('./nn');

const BUCKETS = 2048;
const randomInt = (min, max) => Math.floor(min + Math.random() * (max - min + 1));
const pick = arr => arr[Math.floor(Math.random() * arr.length)];
const sigmoid = z => 1 / (1 + Math.exp(-z));

const words = text => text.toLowerCase().match(/[\p{L}\p{N}']+/gu) || [];
const lines = text => text.split(/[.!?;\n]+/).map(s => s.trim()).filter(s => words(s).length >= 2);

const STOP = new Set(('the a an of to in and is was were are be been on at by for with as it its his her their our your my we they he she ' +
  'from that this these those or but not so if then than into over under about after before every through near while across up out i you do ' +
  'did has have had will would can could just also only very').split(' '));

/* ---------- spelling pieces ---------- */
function hash(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
const gramCache = new Map();
function grams(word) {
  let g = gramCache.get(word);
  if (!g) {
    const w = '<' + word + '>', ids = new Set([hash('w:' + word) % BUCKETS]);
    for (let n = 3; n <= 5; n++) for (let i = 0; i + n <= w.length; i++) ids.add(hash(w.slice(i, i + n)) % BUCKETS);
    g = [...ids]; gramCache.set(word, g);
  }
  return g;
}

/* ---------- layer 1: learn word meanings ---------- */
function trainWordMeanings(sents, opts, onEpoch) {
  const { dim = 32, epochs = 30, window = 3, negatives = 5, lr = 0.08 } = opts;
  const counts = new Map();
  for (const s of sents) for (const w of s) counts.set(w, (counts.get(w) || 0) + 1);
  const vocab = [...counts.keys()], index = new Map(vocab.map((w, i) => [w, i]));
  const freq = vocab.map(w => counts.get(w)), total = freq.reduce((a, b) => a + b, 0);
  const data = sents.map(s => s.map(w => index.get(w)));

  const In = Array.from({ length: BUCKETS }, () => Array.from({ length: dim }, () => (Math.random() - 0.5) / dim));
  const Out = vocab.map(() => new Array(dim).fill(0));

  const weights = freq.map(c => c ** 0.75), sum = weights.reduce((a, b) => a + b, 0), table = [];
  weights.forEach((w, i) => { for (let k = Math.max(1, Math.round((w / sum) * 50000)); k > 0; k--) table.push(i); });
  const keep = freq.map(c => Math.min(1, Math.sqrt(5e-3 / (c / total)) + 5e-3 / (c / total)));
  const vec = new Array(dim), grad = new Array(dim);

  for (let epoch = 0; epoch < epochs; epoch++) {
    const rate = lr * (1 - epoch / epochs) + lr * 0.05;
    let loss = 0, n = 0;
    data.sort(() => Math.random() - 0.5);
    for (const sentence of data) {
      const s = sentence.filter(i => Math.random() < keep[i]);
      for (let p = 0; p < s.length; p++) {
        const g = grams(vocab[s[p]]);
        vec.fill(0);
        for (const b of g) for (let d = 0; d < dim; d++) vec[d] += In[b][d] / g.length;
        grad.fill(0);
        const reach = randomInt(1, window);
        for (let q = Math.max(0, p - reach); q <= Math.min(s.length - 1, p + reach); q++) {
          if (q === p) continue;
          for (let k = -1; k < negatives; k++) {
            const o = k < 0 ? s[q] : table[randomInt(0, table.length - 1)], label = k < 0 ? 1 : 0;
            if (label === 0 && o === s[q]) continue;
            let dot = 0;
            for (let d = 0; d < dim; d++) dot += vec[d] * Out[o][d];
            const pr = sigmoid(dot), step = (label - pr) * rate;
            for (let d = 0; d < dim; d++) { grad[d] += step * Out[o][d]; Out[o][d] += step * vec[d]; }
            loss -= Math.log((label ? pr : 1 - pr) + 1e-9); n++;
          }
        }
        for (const b of g) for (let d = 0; d < dim; d++) In[b][d] += grad[d] / g.length;
      }
    }
    onEpoch && onEpoch(epoch + 1, epochs, loss / Math.max(1, n));
  }
  return makeModel({ dim, vocab, counts: freq, total, In });
}

function makeModel(m) {
  m.index = new Map(m.vocab.map((w, i) => [w, i]));
  m.cache = new Map();
  m.rawCache = new Map();
  m.scale = m.scale || 1;
  m.ref = m.ref || 1;
  m.mu = m.mu || new Array(m.dim).fill(0);
  return m;
}

/** Raw meaning of one word: average of its spelling pieces. Pieces never trained stay tiny, so unseen words fade out. */
function wordRaw(m, word) {
  let v = m.rawCache.get(word);
  if (!v) {
    const g = grams(word);
    v = new Array(m.dim).fill(0);
    for (const b of g) for (let d = 0; d < m.dim; d++) v[d] += m.In[b][d] / g.length;
    m.rawCache.set(word, v);
  }
  return v;
}

/** Unit-length meaning of one word (for comparing words). */
function wordUnit(m, word) {
  let u = m.cache.get(word);
  if (!u) {
    const v = wordRaw(m, word).map((x, d) => x - m.mu[d]), len = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
    u = v.map(x => x / len); m.cache.set(word, u);
  }
  return u;
}

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const cosine = (a, b) => dot(a, b) / ((Math.sqrt(dot(a, a)) * Math.sqrt(dot(b, b))) || 1);

function nearest(m, word, k = 5) {
  const u = wordUnit(m, word.toLowerCase());
  return m.vocab.filter(w => w !== word.toLowerCase()).map(w => ({ w, sim: dot(u, wordUnit(m, w)) }))
    .sort((a, b) => b.sim - a.sim).slice(0, k);
}

/* ---------- layer 2: sentence meaning ---------- */
function sentenceVector(m, text) {
  const v = new Array(m.dim).fill(0), known = [], unknown = [];
  let weight = 0, understood = 0, content = 0;
  for (const w of words(text)) {
    const i = m.index.get(w);
    (i === undefined ? unknown : known).push(w);
    const a = 1e-3 / (1e-3 + (i === undefined ? 1 : m.counts[i]) / m.total), r = wordRaw(m, w);
    for (let d = 0; d < m.dim; d++) v[d] += a * r[d];
    weight += a;
    if (!STOP.has(w)) { content++; understood += Math.min(1, Math.sqrt(dot(r, r)) / m.ref); }   // how well does it know this word?
  }
  if (!weight) return { x: null, known, unknown };
  const x = v.map(t => (m.scale * t) / weight), coverage = content ? understood / content : 0;
  return { x, input: [...x, 3 * coverage], coverage, known, unknown };
}

/** Set the sizes the judge expects: typical sentences get length 3, and "well known word" = typical trained word size. */
function calibrate(m, taught) {
  m.cache.clear();
  m.mu = new Array(m.dim).fill(0);                       // the direction every word shares; removed when comparing meanings
  m.vocab.forEach(w => wordRaw(m, w).forEach((t, d) => { m.mu[d] += t / m.vocab.length; }));
  const norms = m.vocab.map(w => Math.sqrt(dot(wordRaw(m, w), wordRaw(m, w)))).sort((a, b) => a - b);
  m.ref = norms[Math.floor(norms.length / 2)] || 1;
  m.scale = 1;
  const sample = Array.from({ length: Math.min(200, taught.length) }, () => pick(taught));
  const lens = sample.map(s => { const { x } = sentenceVector(m, s.join(' ')); return x ? Math.sqrt(dot(x, x)) : 0; });
  m.scale = 3 / ((lens.reduce((a, b) => a + b, 0) / lens.length) || 1);
}

/* ---------- layer 3: judge same / different ---------- */
const LETTERS = 'abcdefghijklmnopqrstuvwxyz';
const COMMON = ('time person year way day thing man world life hand part child eye woman place work week case point number group problem fact ' +
  'run walk talk think write read open close build break find give take make see look use call try ask need feel leave keep let begin show ' +
  'good new first last long great little own other old right big high different small large next early young important few public bad same').split(' ');

function noisy(ws) {
  const w = ws.slice();
  if (w.length > 3 && Math.random() < 0.3) w.splice(randomInt(0, w.length - 1), 1);
  if (Math.random() < 0.3) {                                                                     // spelling variant: sailors, sailing, a typo
    const i = randomInt(0, w.length - 1), x = w[i];
    w[i] = pick([x + 's', x + 'ed', x + 'ing', x + 'er', x.slice(0, -1), x.length > 3 ? x.slice(0, 2) + x[3] + x[2] + x.slice(4) : x]);
  }
  if (Math.random() < 0.4) for (let k = randomInt(1, 2); k > 0; k--) w.splice(randomInt(0, w.length), 0, pick(COMMON));  // filler words
  return w.join(' ');
}

function buildDataset(m, taught, others, n) {
  const salad = () => Array.from({ length: randomInt(3, 9) }, () => pick(m.vocab)).join(' ');
  const outside = [
    salad, salad,
    () => Array.from({ length: randomInt(3, 8) }, () => pick([...STOP])).join(' '),
    () => Array.from({ length: randomInt(3, 9) }, () => pick(COMMON)).join(' '),
    () => Array.from({ length: randomInt(2, 6) }, () => Array.from({ length: randomInt(3, 9) }, () => pick([...LETTERS])).join('')).join(' '),
  ];
  const data = [];
  while (data.length < n) {
    const same = data.length % 2 === 0;
    const text = same ? noisy(pick(taught)) : others.length && Math.random() < 0.6 ? noisy(pick(others)) : pick(outside)();
    const { input } = sentenceVector(m, text);
    if (input) data.push({ x: input, y: same ? 1 : 0 });
  }
  return data;
}

const newJudge = (dim, hidden = 24) => new Net([dim + 1, hidden, Math.max(6, hidden >> 1), 1]);

/* ---------- understanding what the text is about ---------- */
function contentWords(m, minCount = 2) {
  const list = m.vocab.map((w, i) => ({ w, c: m.counts[i] })).filter(x => !STOP.has(x.w) && x.c >= minCount);
  return list.length >= 6 ? list : m.vocab.map((w, i) => ({ w, c: m.counts[i] })).filter(x => !STOP.has(x.w));
}

/** Group words by meaning (k-means on the learned word vectors). */
function meaningGroups(m, k) {
  const items = contentWords(m), pts = items.map(x => wordUnit(m, x.w));
  k = Math.max(1, Math.min(k, Math.floor(items.length / 3)));
  let centers = [...pts].sort(() => Math.random() - 0.5).slice(0, k), assign = [];
  for (let it = 0; it < 30; it++) {
    assign = pts.map(p => centers.reduce((best, c, j) => (dot(p, c) > dot(p, centers[best]) ? j : best), 0));
    centers = centers.map((c, j) => {
      const mem = pts.filter((_, i) => assign[i] === j);
      if (!mem.length) return c;
      const sum = new Array(m.dim).fill(0); mem.forEach(p => p.forEach((x, d) => { sum[d] += x; }));
      const len = Math.sqrt(dot(sum, sum)) || 1; return sum.map(x => x / len);
    });
  }
  return centers.map((_, j) => items.filter((_, i) => assign[i] === j).sort((a, b) => b.c - a.c))
    .filter(g => g.length >= 2).sort((a, b) => b.length - a.length);
}

function centroid(m, taught) {
  const c = new Array(m.dim).fill(0);
  for (const s of taught) { const { x } = sentenceVector(m, s.join(' ')); if (x) x.forEach((v, d) => { c[d] += v; }); }
  const len = Math.sqrt(dot(c, c)) || 1;
  return c.map(x => x / len);
}

/** For each of the person's own lines, remember its meaning so a question can be matched to the closest one. */
const indexLines = (m, rawLines) => rawLines.map(text => ({ text, x: sentenceVector(m, text).x })).filter(l => l.x);
function closestLine(index, x) {
  let best = null, top = -2;
  for (const l of index) { const s = cosine(x, l.x); if (s > top) { top = s; best = l; } }
  return best && { text: best.text, sim: top };
}

/* ---------- save / load ---------- */
const round = (a, p = 3) => a.map(x => +x.toFixed(p));
const toJSON = (m, net, center, raw) => ({ type: 'text-brain', dim: m.dim, scale: m.scale, ref: m.ref, mu: round(m.mu, 4), vocab: m.vocab, counts: m.counts, total: m.total,
  In: m.In.map(r => round(r)), centroid: round(center, 4), lines: raw.slice(0, 1500), net: net.toJSON() });
function fromJSON(d) {
  if (d.type !== 'text-brain') throw new Error('Not a weights file from this program.');
  const model = makeModel({ dim: d.dim, scale: d.scale, ref: d.ref, mu: d.mu, vocab: d.vocab, counts: d.counts, total: d.total, In: d.In });
  return { model, net: new Net(d.net.sizes, { W: d.net.W, b: d.net.b }), centroid: d.centroid, raw: d.lines, index: indexLines(model, d.lines) };
}

module.exports = { STOP, calibrate, words, lines, trainWordMeanings, nearest, sentenceVector, cosine, buildDataset, newJudge,
  contentWords, meaningGroups, centroid, indexLines, closestLine, toJSON, fromJSON, pick };
