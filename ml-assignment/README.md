# Trigram Language Model

This directory contains the core assignment files for the Trigram Language Model.

## Steps to Run the Code

## 1. Set up the environment
Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate      #mac
venv\Scripts\activate          #windows
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 2. Running the Trigram Language Model (Task 1)

The model uses `example_corpus.txt` by default.

To train the model and generate text:

```bash
python -m src.generate
```

or:

```bash
python src/generate.py
```

## 3. Running Tests

Run the tests from the project root:

```bash
PYTHONPATH=. pytest -q
```

This ensures the `src/` module is discoverable by pytest.

## 4. Running the Optional Attention Task (Task 2)

To run the scaled dot-product attention demo:

```bash
python ml-assignment/attention/demo.py
```

This script loads sample Q, K, and V matrices and prints the attention weights and output.


## Design Choices

Please document your design choices in the `evaluation.md` file. This should be a 1-page summary of the decisions you made and why you made them.
