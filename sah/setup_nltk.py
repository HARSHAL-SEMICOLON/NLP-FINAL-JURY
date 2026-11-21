"""Download the NLTK data packages into ./nltk_data. Run: python -m sah.setup_nltk"""
import nltk

from sah.config import ROOT

target = ROOT / "nltk_data"
target.mkdir(exist_ok=True)
for pkg in ("punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger_eng", "maxent_ne_chunker_tab", "words", "vader_lexicon"):
    ok = nltk.download(pkg, download_dir=str(target), quiet=True)
    print(f"{pkg:12s} {'ok' if ok else 'FAILED (the app falls back to built-in versions)'}")
