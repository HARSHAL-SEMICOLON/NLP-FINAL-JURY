"""Generate sample_data/NLP_Unit1_Notes.pdf – demo material for testing and the viva demo.
Run: python sample_data/make_sample_pdf.py"""
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

PAGES = [
    ("Unit 1 - Introduction to NLP", """
Natural Language Processing (NLP) is a field of artificial intelligence that enables computers to understand, interpret and generate human language. NLP combines linguistics, statistics and machine learning.
Typical NLP applications include machine translation, sentiment analysis, chatbots, spam filtering, question answering and text summarization.
Ambiguity is the main challenge in NLP. Lexical ambiguity occurs when a word has several meanings, for example the word bank. Syntactic ambiguity occurs when a sentence can be parsed in more than one way.
The classical NLP pipeline consists of text cleaning, tokenization, stop-word removal, stemming or lemmatization, feature extraction and modelling.
"""),
    ("Unit 1 - Text Preprocessing", """
Tokenization is the process of splitting text into smaller units called tokens, such as words, subwords or sentences. Word tokenization splits on spaces and punctuation, while subword tokenization such as Byte Pair Encoding splits rare words into frequent pieces.
Stop words are very common words such as the, is and of that carry little meaning on their own. Stop-word removal reduces noise for tasks like keyword search and topic modelling, but it should not be used for tasks where negation matters, such as sentiment analysis.
Stemming is a rule-based process that chops word endings to reach a root form, for example running becomes run and studies becomes studi. The Porter stemmer is the most widely used stemming algorithm.
Lemmatization is the process of reducing a word to its dictionary form called the lemma, using vocabulary and part-of-speech information. Lemmatization of better gives good, which stemming cannot do.
"""),
    ("Unit 1 - Text Representation", """
Bag of Words is a representation that describes a document by the counts of the words it contains, ignoring grammar and word order.
TF-IDF stands for Term Frequency - Inverse Document Frequency. TF-IDF is a weighting scheme that gives high weight to words that are frequent in a document but rare across the whole corpus. The IDF of a term is the logarithm of the total number of documents divided by the number of documents containing the term.
Word embeddings are dense vectors that place words with similar meanings close together in vector space. Word2Vec learns embeddings with the Skip-gram or CBOW objective, and GloVe learns them from global co-occurrence statistics.
Cosine similarity measures the angle between two vectors and is the standard way to compare embeddings. A cosine similarity of 1 means the vectors point in the same direction.
"""),
    ("Unit 1 - NLP versus Large Language Models", """
A Large Language Model (LLM) is a transformer-based neural network with billions of parameters, pretrained on massive text corpora to predict the next token. Examples include GPT, Gemini, Llama and Claude.
Traditional NLP builds a separate pipeline and model for each task, using hand-crafted features such as TF-IDF. LLMs instead learn general language ability during pretraining and are adapted to many tasks through prompting or fine-tuning.
The Transformer is a neural architecture based entirely on self-attention, introduced in the paper Attention Is All You Need in 2017. Self-attention lets every token weigh the importance of every other token in the sequence, which captures long-range context and allows parallel training.
Hallucination refers to an LLM generating fluent but factually incorrect content. Retrieval-Augmented Generation (RAG) reduces hallucination by retrieving relevant passages from a trusted knowledge base and adding them to the prompt before generation.
"""),
]

doc = fitz.open()
for title, body in PAGES:
    page = doc.new_page()
    page.insert_text((50, 60), title, fontsize=16)
    rect = fitz.Rect(50, 90, 545, 800)
    page.insert_textbox(rect, " ".join(body.split("\n")).strip().replace("  ", "\n\n"), fontsize=11)
out = Path(__file__).parent / "NLP_Unit1_Notes.pdf"
doc.save(out)
print("wrote", out)
