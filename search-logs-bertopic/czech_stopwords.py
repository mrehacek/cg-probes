"""Czech stopword list for c-TF-IDF vectorization in the BERTopic step.

Combines `stopwordsiso` ('cs') with a small set of search-UX tokens that show
up in the top of the aggregated CSV but carry no topical signal.
"""

from __future__ import annotations

# Tokens that sneak in from search UX and aren't real Czech stopwords.
EXTRA_STOPWORDS_CS: frozenset[str] = frozenset({
    "cz", "www", "http", "https", "html", "php",
    "the", "and", "for", "with",  # stray English fragments in top queries
})


def czech_stopwords() -> list[str]:
    """Return the merged stopword list used by the CountVectorizer."""
    import stopwordsiso
    return sorted(set(stopwordsiso.stopwords("cs")) | EXTRA_STOPWORDS_CS)
