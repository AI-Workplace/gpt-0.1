# Text Brain

One neural network, in pure JavaScript, that runs in the terminal. No libraries, no install. Needs Node 14+.

You train it on **one text**. It learns what the words mean and writes the weights to `brain.json`. Then you type anything and it tells you whether it means the same kind of thing as your text, and why.

## Quick start

    node cli.js learn examples/sea.txt
    node cli.js ask "The sailor steered the boat through the storm"
    node cli.js meaning
    node cli.js similar ship
    node cli.js                      # interactive mode

## Commands

    node cli.js learn <file> [--not <file>] [--dim 32] [--epochs 30] [--out brain.json]
    node cli.js learn --text "words you want it to learn"
    node cli.js ask "text to check" [--weights brain.json]
    node cli.js meaning              what the text is about, words grouped by meaning
    node cli.js similar <word>       words closest in meaning

Interactive mode: type text to ask, or `/meaning`, `/similar <word>`, `/learn <file>`, `/teach <text>`, `/not <file>`, `/save`, `/load`, `/help`, `/exit`.

## What `ask` tells you

    P = 0.9951     SAME ✓   understood 83% of its words
    closest idea in your text: "The skipper guided the yacht across the bay"
    how it read your words: vessel ≈ yacht  ocean ≈ island
    never seen, read by spelling: crossed

- **P**: the probability it means the same kind of thing (P ≥ 0.5 = SAME)
- **understood**: how well it knows the words (unknown topics score low)
- **closest idea**: the line of your text nearest in meaning
- **how it read your words**: each word's nearest neighbour in what it learned

## How it works

1. **Word meanings.** Each word is the average of its spelling pieces (3 to 5 letter chunks) looked up in a learned table. The table is trained by predicting the words around each word: `P(neighbour | word) = sigmoid(vec(word) · out[neighbour])`. Real neighbours are pushed up, random words down. Words used in similar places get similar numbers (ship ≈ boat ≈ vessel). Because of the spelling pieces, words it never saw (sailors, sailing, typos) still get a sensible meaning.
2. **Sentence meaning.** The average of its word meanings, with rare words counting more than common ones.
3. **The judge.** A small network (tanh hidden layers, sigmoid output, cross-entropy loss, backpropagation, gradient descent) takes the sentence meaning plus how much of the sentence it understands, and outputs P. Its training examples are made for you: pieces of your text with dropped words, typos and filler (label 1), versus word salad, function words only, random strings and unrelated common words (label 0).
4. **Understanding the text.** Word meanings are clustered (k-means) into groups, and the most important words are listed (`meaning`).

## Honest limits

- It only knows what is in your text. Words and topics it has never seen are marked as not understood and score DIFFERENT.
- More text means better meanings. A few hundred sentences gives clear word groups. About 14 sentences works for same/different but word meanings are rough (the program warns you).
- Everything it sees is about your text alone, so it cannot tell two similar-looking topics apart unless you add `--not` texts to give contrast.
- "Meaning" means used in similar contexts. It does not understand grammar, negation or facts.

## Files

- `nn.js` the judge network and training maths
- `brain.js` word meanings, sentence meaning, training examples, grouping
- `cli.js` terminal commands and interactive mode
- `examples/` `sea.txt` (400 sentences) and `story.txt` (14 sentences), plus themed texts and `make-corpus.js`
