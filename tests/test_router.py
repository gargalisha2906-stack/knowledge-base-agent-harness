from kbharness.router import DOMAINS, KeywordRouter

router = KeywordRouter(DOMAINS, min_confidence=0.35)


def test_clear_domain_query():
    decision = router.route("How do I get a refund?")
    assert decision.domain == "refunds"
    assert decision.doc_ids == ["refund_policy.md"]


def test_unknown_query_falls_to_other():
    decision = router.route("What is the weather today?")
    assert decision.domain == "other"
    # "other" may search every document
    assert len(decision.doc_ids) == sum(len(d.doc_ids) for d in DOMAINS)


def test_single_word_keywords_match_whole_tokens_only():
    # "taxes" must not fire the "tax" keyword, "databases" must not fire "data"
    assert router.route("Can you help me with my taxes?").domain != "billing"
    assert router.route("Write me a poem about databases").domain != "account"
