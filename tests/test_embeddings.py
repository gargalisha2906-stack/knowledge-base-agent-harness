from kbharness.embeddings import HashingEmbedder, cosine_similarity, tokenize


def test_stopwords_removed():
    assert tokenize("How do I get a refund?") == ["get", "refund"]


def test_similar_texts_score_higher():
    e = HashingEmbedder()
    close = cosine_similarity(e.embed("refund policy for returns"),
                              e.embed("how do I return an item for a refund"))
    far = cosine_similarity(e.embed("refund policy for returns"),
                            e.embed("kubernetes cluster autoscaling"))
    assert close > far


def test_inflections_match_via_trigrams():
    e = HashingEmbedder()
    sim = cosine_similarity(e.embed("refunding"), e.embed("refund"))
    assert sim > 0.35  # trigram overlap keeps inflections close (exact value depends on hash collisions)


def test_empty_text_zero_vector():
    e = HashingEmbedder()
    assert cosine_similarity(e.embed(""), e.embed("refund")) == 0.0
