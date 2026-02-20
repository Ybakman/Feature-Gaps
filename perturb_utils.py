import random
import re


def attack_coherence_sentence_order(text: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= 1:
        return text
    random.shuffle(sentences)
    return " ".join(sentences)


def mask_text(text: str, percent: int = 15) -> str:
    words = text.split()
    if not words:
        return text
    n_mask = max(1, int(len(words) * (percent / 100.0)))
    idxs = random.sample(range(len(words)), min(n_mask, len(words)))
    for i in idxs:
        words[i] = "[MASK]"
    return " ".join(words)


def attack_grammar_typos(text: str) -> str:
    return text.replace(" the ", " teh ")


def attack_grammar_word_order(text: str) -> str:
    words = text.split()
    if len(words) < 4:
        return text
    i = random.randint(0, len(words) - 2)
    words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)
