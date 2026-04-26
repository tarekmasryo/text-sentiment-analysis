from __future__ import annotations

import html
import re

from sklearn.base import BaseEstimator, TransformerMixin


class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._clean(text) for text in X]

    def _clean(self, text) -> str:
        text = "" if text is None else str(text)
        text = html.unescape(text)
        text = text.replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ")
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[\\r\\n\\t]+", " ", text)
        text = text.lower() if self.lowercase else text

        replacements = {
            "won't": "will not",
            "can't": "can not",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
            "it's": "it is",
            "that's": "that is",
            "there's": "there is",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(r"[^a-zA-Z\\s]", " ", text)
        text = re.sub(r"\\s+", " ", text).strip()
        return text
