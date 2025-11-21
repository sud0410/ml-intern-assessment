# Evaluation

This task told me to make a simple trigram language model. A trigram model looks at the last two words and tries to guess what the next word will be. It doesn't get what things mean; it just follows patterns in the text.

## Tasks

### 1. Used the Trigram Model

I filled in the empty TrigramModel by adding:

- Code to make the text cleaner (lowercase and remove punctuation)
- Code that breaks up the  text into words and sentences
- Tokens like `<s>` and `</s>` that add space between sentences so the model knows where they start and end
- Logic for counting:
  - one word (unigram)
  - pairs of words (bigrams)
  - three-word combinations (trigrams)
- A function that chooses the next word based on the last two words

### 2. Used Simple Probability-Based Word Choice and Added Backoff

I used `random.choices()` instead of always picking the same word. This way, the model picks the next word based on how often it showed up in the  training text. This means that the text that is made is always different. If the model doesn't know a trigram, it tries a  bigram.

If that doesn't work either, it goes back to picking from all the words.

This stops the generator from crashing when it sees patterns it doesn't know about.

### 3. Fixed the Generate Script and Fixed the Test Import Issue

At first, `generate.py` couldn't find the model or the data file.

I made it work like this:

- It works  whether I run `python src/generate.py` or `python -m src.generate`. No matter where I run the command from, it always finds the `example_corpus.txt` file and I ran tests with `PYTHONPATH=. pytest -q` because Pytest couldn't find the src folder.

After that, all the tests worked.

## Simple Model Explanation

- It reads the example text.
- It learns which words usually follow which pairs of words.
- When generating, it  starts with `<s>` `<s>` and keeps picking one word at a time.
- It stops when it hits `</s>` or reaches the max length.

**Example output:**

it contains a few sentences to get you started

---

## Optional Task 2 — Scaled Dot-Product Attention

I added a small NumPy-only implementation of scaled dot-product attention and a demo to show it working.

- It uses dot products to find the similarity between queries (Q) and keys (K).
- To keep the numbers stable, it scales the scores by 1/sqrt(d_k).
-  uses a mask to block some positions.
- Uses softmax to  get attention weights, which add up to 1 for each query.
- To get the final output, it multiplies the weights by the values (V).

### Why This Is Helpful

It lets the model look at all the input positions and decide how much each one should add to the output for each query position.

### Demo Output

```
Attention Weights:
[[[0.76036844 0.23963156]]]
Output:
[[[1.52073688 0.47926312]]]
```

This shows the attention weights (how much the single query paid attention to each key) and the weighted sum that came from V.

---

## Conclusion

This finishes Task 1 and Task 2. The trigram model is simple and easy to understand. The attention demo shows how modern transformers work with basic math.