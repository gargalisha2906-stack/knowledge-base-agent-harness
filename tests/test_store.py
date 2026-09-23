from kbharness.chunking import Chunk
from kbharness.embeddings import HashingEmbedder
from kbharness.store import VectorStore


def make_store():
    store = VectorStore(HashingEmbedder())
    store.add(Chunk("a.md#0", "a.md", 0, "refund policy for yearly plans"))
    store.add(Chunk("b.md#0", "b.md", 0, "webhook delivery retries"))
    return store


def test_search_ranks_relevant_first():
    results = make_store().search("how do refunds work", top_k=1)
    assert results[0].chunk.doc_id == "a.md"


def test_domain_filter():
    results = make_store().search("refund", top_k=5, doc_ids={"b.md"})
    assert all(r.chunk.doc_id == "b.md" for r in results)


def test_roundtrip(tmp_path):
    store = make_store()
    path = str(tmp_path / "store.json")
    store.save(path)
    fresh = VectorStore(HashingEmbedder())
    fresh.load(path)
    assert len(fresh) == 2
    assert fresh.search("refund", top_k=1)[0].chunk.id == "a.md#0"
