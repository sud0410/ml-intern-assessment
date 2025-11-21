import random
from pathlib import Path

try:
    from src.ngram_model import TrigramModel
except ModuleNotFoundError:
    from ngram_model import TrigramModel

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "example_corpus.txt"

def main():
    model = TrigramModel()
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    
    model.fit(text)
    generated_text = model.generate()
    
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()