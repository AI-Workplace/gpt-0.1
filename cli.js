#!/usr/bin/env node
'use strict';
const fs = require('fs');
const readline = require('readline');
const B = require('./brain');

/* ---------- terminal helpers ---------- */
const useColor = process.stdout.isTTY && !process.env.NO_COLOR;
const paint = code => s => (useColor ? `\x1b[${code}m${s}\x1b[0m` : String(s));
const [bold, dim, red, green, cyan] = [1, 2, 31, 32, 36].map(paint);
const spark = v => { const t = Math.max(...v, 1e-9); return v.map(x => '▁▂▃▄▅▆▇█'[Math.min(7, Math.floor((x / t) * 7.999))]).join(''); };
const bar = (p, w = 30) => '█'.repeat(Math.round(p * w)) + '░'.repeat(w - Math.round(p * w));
const live = line => (process.stdout.isTTY ? process.stdout.write('\r' + line + '   ') : console.log(line));

const DEFAULT_FILE = 'brain.json';
const state = { taught: [], others: [], opts: { dim: 32, epochs: 30 }, steps: 2000, lr: 0.2, out: DEFAULT_FILE,
  model: null, net: null, center: null, raw: [], index: [] };
const readText = f => { try { return fs.readFileSync(f, 'utf8'); } catch { throw new Error(`Cannot read file: ${f}`); } };
const need = () => { if (!state.model) throw new Error('No weights yet. Run:  node cli.js learn yourfile.txt'); };

/* ---------- learn ---------- */
function learn() {
  if (!state.taught.length) throw new Error('Nothing to learn. Give it a text file or --text "...".');
  const taughtL = state.taught.flatMap(B.lines), otherL = state.others.flatMap(B.lines);
  const taughtS = taughtL.map(B.words), otherS = otherL.map(B.words);
  console.log(`\n${bold('Step 1: learning what words mean')}  ${dim(`${taughtS.length + otherS.length} sentences from your text`)}`);

  const hist = [];
  const model = B.trainWordMeanings([...taughtS, ...otherS], state.opts, (e, n, loss) => {
    hist.push(loss);
    if (e % 2 === 0 || e === n) live(`epoch ${String(e).padStart(3)}/${n}  loss ${loss.toFixed(4)}  ${cyan(spark(hist))}`);
  });
  if (process.stdout.isTTY) process.stdout.write('\n');

  B.calibrate(model, taughtS);
  console.log(`\n${bold('Step 2: learning what "same meaning" looks like')}`);
  const net = B.newJudge(model.dim), data = B.buildDataset(model, taughtS, otherS, 800), check = B.buildDataset(model, taughtS, otherS, 200);
  const h2 = [], every = Math.max(1, Math.floor(state.steps / 30));
  for (let s = 1; s <= state.steps; s++) {
    net.train(Array.from({ length: 32 }, () => B.pick(data)), state.lr);
    if (s % every === 0 || s === state.steps) {
      let loss = 0, ok = 0;
      for (const { x, y } of check) { const p = net.predict(x); loss -= y ? Math.log(p + 1e-7) : Math.log(1 - p + 1e-7); if ((p > 0.5) === !!y) ok++; }
      h2.push(loss / check.length);
      live(`step ${String(s).padStart(5)}/${state.steps}  loss ${(loss / check.length).toFixed(4)}  acc ${(ok / check.length * 100).toFixed(1).padStart(5)}%  ${cyan(spark(h2))}`);
    }
  }
  if (process.stdout.isTTY) process.stdout.write('\n');

  state.raw = taughtL;
  Object.assign(state, { model, net, center: B.centroid(model, taughtS), index: B.indexLines(model, taughtL) });
  console.log(`\n${green('Done.')} ${model.vocab.length} words learned from ${taughtS.length + otherS.length} sentences.`);
  if (taughtS.length + otherS.length < 100) console.log(dim('  Tip: with little text the word meanings are rough. More text gives better meaning.'));
  meaning();
  save(state.out);
}

/* ---------- what does the text mean? ---------- */
function meaning() {
  need();
  const m = state.model, top = B.contentWords(m).sort((a, b) => b.c - a.c).slice(0, 8);
  console.log(`\n${bold('What your text is about')}  ${dim('(most important words)')}`);
  console.log('  ' + top.map(x => x.w).join(', '));
  const groups = B.meaningGroups(m, 5);
  if (groups.length) {
    console.log(`\n${bold('Words it grouped by meaning')}`);
    groups.forEach((g, i) => console.log(`  ${cyan(`group ${i + 1}`)}  ${g.slice(0, 8).map(x => x.w).join(', ')}`));
  }
  console.log('');
}

/* ---------- save / load ---------- */
function save(file) {
  fs.writeFileSync(file, JSON.stringify(B.toJSON(state.model, state.net, state.center, state.raw)));
  console.log(`Weights saved to ${bold(file)}  (${state.model.vocab.length} words, ${state.model.dim} numbers each)`);
}
function load(file) {
  const r = B.fromJSON(JSON.parse(readText(file)));
  Object.assign(state, { model: r.model, net: r.net, center: r.centroid, raw: r.raw, index: r.index });
}

/* ---------- ask / similar ---------- */
function ask(text) {
  need();
  if (!text.trim()) return;
  const { x, input, coverage, known, unknown } = B.sentenceVector(state.model, text);
  if (!input) { console.log(`\n  ${red(bold('DIFFERENT ✗'))}  ${dim('no words to read')}\n`); return; }
  const p = state.net.predict(input), same = p >= 0.5, near = B.closestLine(state.index, x);
  console.log(`\n  ${bar(p)}  P = ${p.toFixed(4)}`);
  console.log(`  ${same ? green(bold('SAME ✓')) : red(bold('DIFFERENT ✗'))}  ${dim(`understood ${Math.round(coverage * 100)}% of its words`)}`);
  if (near) console.log(`  ${dim('closest idea in your text:')} “${near.text}” ${dim(`(${near.sim.toFixed(2)})`)}`);
  const seen = [...new Set([...known, ...unknown])].filter(w => !B.STOP.has(w)).slice(0, 6);
  if (seen.length) console.log('  ' + dim('how it read your words: ') + seen.map(w => `${w} ${dim('≈')} ${B.nearest(state.model, w, 1)[0].w}`).join('  '));
  if (unknown.length) console.log('  ' + dim(`never seen, read by spelling: ${[...new Set(unknown)].join(', ')}`));
  console.log('');
}
function similar(word) {
  need();
  if (!word) throw new Error('Give a word, e.g.  similar ship');
  console.log(`\n  words closest in meaning to ${bold(word)}:`);
  B.nearest(state.model, word, 8).forEach(({ w, sim }) => console.log(`  ${bar(Math.max(0, sim), 20)} ${sim.toFixed(2)}  ${w}`));
  console.log('');
}

/* ---------- interactive shell ---------- */
const HELP = `
  ${bold('Just type text')} to ask: does it mean the same kind of thing as what I was taught?

  /meaning          what the text is about          /similar <word>   words close in meaning
  /learn <file>     learn from a file               /teach <text>     learn from text typed here
  /not <file>       add a text with a different meaning (then /learn)
  /save [file]      /load [file]                    /help    /exit
`;
function shell() {
  console.log(`\n${bold('Text Brain')}  ${dim('— one neural network that learns what your text means')}`);
  if (fs.existsSync(DEFAULT_FILE)) { try { load(DEFAULT_FILE); console.log(green(`Loaded ${DEFAULT_FILE}.`)); } catch {} }
  console.log(HELP);
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: bold('text> ') });
  rl.prompt();
  rl.on('line', line => {
    try {
      const [cmd, ...r] = line.trim().split(/\s+/), arg = r.join(' ');
      if (!line.trim()) { /* skip */ }
      else if (cmd === '/exit' || cmd === '/quit') return rl.close();
      else if (cmd === '/help') console.log(HELP);
      else if (cmd === '/meaning') meaning();
      else if (cmd === '/similar') similar(arg);
      else if (cmd === '/learn') { if (arg) state.taught = [readText(arg)]; learn(); }
      else if (cmd === '/teach') { state.taught = [arg]; learn(); }
      else if (cmd === '/not') { state.others.push(readText(arg)); console.log(`Added ${arg}. Now /learn.`); }
      else if (cmd === '/save') { need(); save(arg || DEFAULT_FILE); }
      else if (cmd === '/load') { load(arg || DEFAULT_FILE); console.log(green('Weights loaded.')); }
      else ask(line);
    } catch (e) { console.log(red('  ' + e.message)); }
    rl.prompt();
  });
  rl.on('close', () => { console.log('bye'); process.exit(0); });
}

/* ---------- command line ---------- */
function parseArgs(argv) {
  const flags = { not: [], text: [] }, pos = [];
  for (let i = 0; i < argv.length; i++) {
    if (!argv[i].startsWith('--')) { pos.push(argv[i]); continue; }
    const k = argv[i].slice(2), v = argv[++i];
    if (k === 'not' || k === 'text') flags[k].push(v); else flags[k] = v;
  }
  return { flags, pos };
}
const USAGE = `
${bold('Text Brain')} — train one neural network on a text, then ask if new text means the same thing.

  node cli.js learn <file> [--not <file>] [--dim 32] [--epochs 30] [--out brain.json]
  node cli.js learn --text "words you want it to learn"
  node cli.js ask "some text to check" [--weights brain.json]
  node cli.js meaning                    what the text is about, words grouped by meaning
  node cli.js similar <word>             words closest in meaning
  node cli.js                            interactive mode
`;
try {
  const [command, ...rest] = process.argv.slice(2), { flags, pos } = parseArgs(rest);
  if (flags.dim) state.opts.dim = +flags.dim;
  if (flags.epochs) state.opts.epochs = +flags.epochs;
  if (flags.out) state.out = flags.out;
  if (command === 'learn') { state.taught = [...pos.map(readText), ...flags.text]; state.others = flags.not.map(readText); learn(); }
  else if (command === 'ask') { load(flags.weights || DEFAULT_FILE); pos.length ? ask(pos.join(' ')) : shell(); }
  else if (command === 'meaning') { load(flags.weights || DEFAULT_FILE); meaning(); }
  else if (command === 'similar') { load(flags.weights || DEFAULT_FILE); similar(pos[0]); }
  else if (command === 'help' || command === '--help') console.log(USAGE);
  else shell();
} catch (e) { console.error(red('Error: ' + e.message)); process.exit(1); }
