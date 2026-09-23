from kbharness.chunking import chunk_document, chunk_text


def test_overlap_step():
    # step = chunk_size - overlap, so windows share `overlap` characters
    chunks = chunk_text("a" * 1000, chunk_size=400, overlap=80)
    assert len(chunks) == 3
    assert chunks[0][-80:] == chunks[1][:80]


def test_short_text_one_chunk():
    assert chunk_text("hello", chunk_size=400, overlap=80) == ["hello"]


def test_empty_text():
    assert chunk_text("   ", chunk_size=400, overlap=80) == []


def test_overlap_must_be_smaller():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_chunk_document_ids():
    chunks = chunk_document("doc.md", "x" * 900, 400, 80)
    assert chunks[0].id == "doc.md#0"
    assert chunks[-1].doc_id == "doc.md"
