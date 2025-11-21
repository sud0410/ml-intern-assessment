import re
import random
from collections import defaultdict, Counter

class TrigramModel:
    def __init__(self):
        """
        Trigram model with:
          - self.trigram_counts[(w1, w2)] -> Counter of w3
          - self.bigram_counts[w1] -> Counter of w2
          - self.unigram_counts -> Counter of words
        """
        self.trigram_counts = defaultdict(Counter)
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()

    def _tokenize_text(self, text):
        """
        Split text into sentences, then into word tokens.
        Lowercases text and extracts word-like tokens.
        Returns list of sentences where each sentence is list of tokens.
        """
        text = text.replace("\r", " ").strip()
        
        sentences = re.split(r'[.!?]+', text)
        tokenized = []
        for sent in sentences:
            s = sent.strip().lower()
            if not s:
                continue
            words = re.findall(r"\b\w+\b", s)
            if words:
                tokenized.append(words)
        return tokenized

    #training 
    def fit(self, text):
        """
        Train trigram counts from the raw text.
        """
        tokenized_sentences = self._tokenize_text(text)

        for words in tokenized_sentences:
            padded = ['<s>', '<s>'] + words + ['</s>']

            for w in padded:
                self.unigram_counts[w] += 1
                self.vocab.add(w)

            for i in range(len(padded) - 1):
                w1 = padded[i]
                w2 = padded[i + 1]
                self.bigram_counts[w1][w2] += 1

            for i in range(len(padded) - 2):
                w1 = padded[i]
                w2 = padded[i + 1]
                w3 = padded[i + 2]
                self.trigram_counts[(w1, w2)][w3] += 1

    # sampling helper 
    def _sample_from_counter(self, counter):
        """
        Given a Counter mapping item -> count, sample one item
        proportionally to its count using random.choices.
        """
        if not counter:
            return None
        items = list(counter.keys())
        weights = [counter[i] for i in items]
        return random.choices(items, weights=weights, k=1)[0]

    # generation
    def generate(self, max_length=50):
        """
        Generate text by starting with two start tokens and sampling
        next words using trigram probabilities with backoff to bigram/unigram.
        """
        w1, w2 = '<s>', '<s>'
        generated = []

        for _ in range(max_length):
            trigram_key = (w1, w2)
            if trigram_key in self.trigram_counts and self.trigram_counts[trigram_key]:
                next_word = self._sample_from_counter(self.trigram_counts[trigram_key])
            # backoff to bigram
            elif w2 in self.bigram_counts and self.bigram_counts[w2]:
                next_word = self._sample_from_counter(self.bigram_counts[w2])
            # Backoff to unigram
            else:
                next_word = self._sample_from_counter(self.unigram_counts)
            if next_word is None:
                break
            if next_word == '</s>':
                break
            generated.append(next_word)
            w1, w2 = w2, next_word
        return ' '.join(generated)
