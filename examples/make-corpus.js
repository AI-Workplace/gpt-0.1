// Generates four small themed texts so the network has real meaning to learn.
// Words inside a theme are used in the same kinds of sentences (ship / boat / vessel),
// so the network learns they mean similar things.  Run: node examples/make-corpus.js
const fs = require('fs'), path = require('path');
const pick = a => a[Math.floor(Math.random() * a.length)];
const themes = {
  sea: { slots: { who: ['captain', 'sailor', 'crew', 'skipper', 'fisherman', 'pirate'], craft: ['ship', 'boat', 'vessel', 'yacht', 'schooner'],
      place: ['ocean', 'sea', 'harbour', 'shore', 'island', 'bay'], gear: ['sails', 'mast', 'anchor', 'deck', 'rope', 'compass'],
      weather: ['waves', 'tide', 'storm', 'wind', 'current', 'fog'], verb: ['steered', 'sailed', 'guided', 'rowed', 'pushed'] },
    lines: ['The {who} {verb} the {craft} across the {place}.', 'A {who} watched the {weather} from the {craft}.', 'The {craft} rocked in the {weather} near the {place}.',
      'Every {who} checked the {gear} before the {weather} arrived.', 'The {who} tied the {gear} to the {craft} at the {place}.', 'A strong {weather} pushed the {craft} toward the {place}.'] },
  computers: { slots: { who: ['programmer', 'engineer', 'developer', 'admin', 'student', 'analyst'], device: ['computer', 'laptop', 'server', 'machine', 'desktop'],
      thing: ['code', 'software', 'database', 'network', 'file', 'program'], problem: ['bug', 'error', 'crash', 'virus', 'glitch'], verb: ['fixed', 'updated', 'installed', 'tested', 'restarted', 'debugged'],
      place: ['office', 'lab', 'cloud', 'datacenter'] },
    lines: ['The {who} {verb} the {thing} on the {device}.', 'A {problem} stopped the {device} in the {place}.', 'The {who} found a {problem} in the {thing}.',
      'Every {who} {verb} the {device} after the {problem}.', 'The {thing} on the {device} crashed because of a {problem}.', 'A {who} in the {place} {verb} the {thing}.'] },
  food: { slots: { who: ['chef', 'cook', 'baker', 'waiter', 'mother', 'grandmother'], dish: ['soup', 'pasta', 'bread', 'cake', 'salad', 'pizza'],
      ingredient: ['flour', 'sugar', 'salt', 'butter', 'tomato', 'cheese'], verb: ['baked', 'cooked', 'stirred', 'tasted', 'served', 'chopped'], place: ['kitchen', 'restaurant', 'oven', 'bakery'] },
    lines: ['The {who} {verb} the {dish} in the {place}.', 'A {who} added {ingredient} to the {dish}.', 'The {dish} needs {ingredient} and {ingredient}.',
      'Every {who} in the {place} {verb} the {dish} with {ingredient}.', 'The {who} tasted the {dish} before dinner.', 'A warm {dish} came out of the {place}.'] },
  animals: { slots: { animal: ['fox', 'dog', 'cat', 'horse', 'rabbit', 'wolf'], verb: ['ran', 'jumped', 'chased', 'hunted', 'slept', 'watched'],
      place: ['forest', 'meadow', 'field', 'barn', 'river', 'hill'], thing: ['grass', 'tree', 'burrow', 'nest', 'fence', 'mouse'] },
    lines: ['The {animal} {verb} through the {place}.', 'A {animal} {verb} near the {thing}.', 'The {animal} hid in the {thing} by the {place}.',
      'Every {animal} in the {place} {verb} at dawn.', 'The {animal} {verb} the {thing} in the {place}.', 'A hungry {animal} found a {thing} at the {place}.'] },
};
const fill = ({ slots, lines }) => pick(lines).replace(/\{(\w+)\}/g, (_, k) => pick(slots[k]));
for (const [name, t] of Object.entries(themes)) {
  const text = Array.from({ length: 400 }, () => fill(t)).join('\n') + '\n';
  fs.writeFileSync(path.join(__dirname, name + '.txt'), text);
}
console.log('wrote sea.txt, computers.txt, food.txt, animals.txt');
