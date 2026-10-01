"""Text-cleaning helpers shared by the notebook and the saved pipeline.

Kept in a separate module (not inside the notebook) so that the pickled
pipeline in models/ can be re-loaded in any new Python session.
"""
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

for _resource in ("stopwords", "punkt", "punkt_tab"):
    nltk.download(_resource, quiet=True)

# NLTK's stopword list contains negations. Removing them turns
# "not happy" into "happy" and flips the meaning of the review.
NEGATIONS = {"no", "nor", "not", "don", "t"}


def get_stopwords(keep_negations=True):
    """English stopwords, optionally without the negation words."""
    words = set(stopwords.words("english"))
    return words - NEGATIONS if keep_negations else words


def clean_text(text, keep_negations=True):
    """Lower-case, strip URLs/@mentions/#, drop punctuation, tokenise,
    remove stopwords and return the remaining words as one string."""
    stop_words = get_stopwords(keep_negations)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)


def clean_series(texts, keep_negations=True):
    """Apply clean_text to an iterable of reviews (used inside the sklearn pipeline)."""
    return [clean_text(t, keep_negations) for t in texts]
