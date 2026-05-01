import re
from collections import Counter
from typing import List, Dict

import nltk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUTPUT_FILES, TOP_N_WORDS, MIN_WORD_LEN

nltk.download("stopwords", quiet=True)

try:
    from nltk.corpus import stopwords as _sw
    _UK_BASE = set(_sw.words("ukrainian"))
except Exception:
    _UK_BASE = set()

#  Extended Ukrainian stopword list
UK_STOPWORDS = _UK_BASE | {
    "це", "але", "або", "та", "що", "як", "він", "вона", "вони", "ми",
    "ви", "я", "мене", "тебе", "його", "її", "нас", "вас", "їх",
    "за", "від", "до", "на", "з", "по", "у", "в", "про", "при",
    "для", "між", "через", "після", "перед", "під", "над", "так",
    "не", "же", "бо", "то", "коли", "якщо", "тому", "тоді",
    "який", "яка", "яке", "яких", "свій", "своя", "своє", "своїх",
    "цей", "ця", "цих", "цими", "всі", "все", "тільки", "також",
    "може", "можна", "тобто", "однак", "проте", "адже", "навіть",
    "вже", "ще", "саме", "більш", "менш", "дуже", "досить",
}


#  Tokenization

def tokenize(text: str) -> List[str]:
    words = re.findall(r'\b[а-яіїєґА-ЯІЇЄҐA-Za-z]{' + str(MIN_WORD_LEN) + r',}\b', text.lower())
    return [w for w in words if w not in UK_STOPWORDS and not w.isdigit()]


#  Frequency analysis

def word_frequency(tokens: List[str], top_n: int = TOP_N_WORDS) -> pd.DataFrame:
    counter = Counter(tokens)
    top = counter.most_common(top_n)
    df = pd.DataFrame(top, columns=["word", "count"])
    df["frequency_pct"] = (df["count"] / max(len(tokens), 1) * 100).round(2)
    return df


#  Text statistics

def text_stats(text: str, tokens: List[str]) -> Dict:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]
    return {
        "total_chars":      len(text),
        "total_words":      len(text.split()),
        "unique_words":     len(set(tokens)),
        "total_tokens":     len(tokens),
        "sentences":        len(sentences),
        "type_token_ratio": round(len(set(tokens)) / max(len(tokens), 1), 3),
        "avg_word_len":     round(float(np.mean([len(w) for w in tokens])), 2) if tokens else 0,
        "lexical_density":  round(len(tokens) / max(len(text.split()), 1), 3),
    }


#  Charts

def save_freq_chart(freq_df: pd.DataFrame, path) -> None:
    n = len(freq_df)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, n))
    fig, ax = plt.subplots(figsize=(14, max(6, n * 0.35)))
    bars = ax.barh(freq_df["word"][::-1], freq_df["count"][::-1], color=colors[::-1])
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlabel("Кількість входжень")
    ax.set_title(f"Частотні характеристики тексту — Топ {n} слів")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {path}")


def save_wordcloud(tokens: List[str], path) -> None:
    try:
        from wordcloud import WordCloud
        wc = WordCloud(
            width=1200, height=600,
            background_color="white",
            max_words=100,
            collocations=False,
        ).generate(" ".join(tokens))
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Word Cloud — Найчастіші слова тексту")
        plt.tight_layout()
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved {path}")
    except ImportError:
        print("[WARN] wordcloud not installed — skipping")


#  Annotation (extractive)

def annotate(text: str, tokens: List[str]) -> str:
    # translate to English for LexRank
    try:
        from deep_translator import GoogleTranslator
        snippet = " ".join(text.split()[:300])
        text_en = GoogleTranslator(source="uk", target="en").translate(snippet)
        print("[INFO] Annotation: translated to English for summarization")
    except Exception as e:
        print(f"[WARN] Translation failed: {e} — using original text")
        text_en = " ".join(text.split()[:300])

    # extractive summarization via sumy LexRank
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lex_rank import LexRankSummarizer

        parser     = PlaintextParser.from_string(text_en, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        sentences  = summarizer(parser.document, sentences_count=1)
        summary    = str(sentences[0]).strip() if sentences else ""
        if summary:
            try:
                summary = GoogleTranslator(source="en", target="uk").translate(summary)
                print(f"[INFO] Annotation translated back to Ukrainian")
            except Exception as e:
                print(f"[WARN] Back-translation failed: {e}")
            print(f"[OK] Annotation (LexRank): {len(summary.split())} words")
            return summary
    except Exception as e:
        print(f"[WARN] sumy LexRank failed: {e}")

    # fallback top content words
    freq  = Counter(tokens)
    words = sorted(set(tokens), key=lambda w: -freq[w])[:12]
    return "Key topics: " + ", ".join(words) + "."
