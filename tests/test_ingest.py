"""Tests for ingestion: front-matter, sections, chunking, directory ingest."""

import itertools

import pytest

from supportbot.ingest import (
    chunk_document,
    chunk_text,
    ingest_directory,
    ingest_markdown,
    parse_front_matter,
    split_sections,
)


def test_parse_front_matter_extracts_title():
    meta, body = parse_front_matter("---\ntitle: Hello\ntags: a, b\n---\nBody text")
    assert meta == {"title": "Hello", "tags": "a, b"}
    assert body == "Body text"


def test_parse_front_matter_absent_returns_empty():
    meta, body = parse_front_matter("# Just a heading\ntext")
    assert meta == {}
    assert body.startswith("# Just a heading")


def test_split_sections_groups_by_heading():
    sections = split_sections("# A\nfoo\n## B\nbar\nbaz")
    assert sections[0] == ("A", "foo")
    assert sections[1] == ("B", "bar\nbaz")


def test_split_sections_keeps_leading_text():
    sections = split_sections("intro line\n# A\nfoo")
    assert sections[0] == ("", "intro line")


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_short_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_chunk_text_respects_max_chars_and_overlaps():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, max_chars=60, overlap=20)
    assert len(chunks) > 2
    assert all(len(c) <= 60 for c in chunks)
    # overlap: consecutive chunks share words
    for a, b in itertools.pairwise(chunks):
        assert set(a.split()) & set(b.split())


def test_chunk_text_validation():
    with pytest.raises(ValueError):
        chunk_text("x", max_chars=0)
    with pytest.raises(ValueError):
        chunk_text("x", max_chars=10, overlap=10)


def test_ingest_markdown_uses_front_matter_title():
    doc = ingest_markdown("d1", "---\ntitle: My Title\n---\n# S\nbody", source="f.md")
    assert doc.title == "My Title"
    assert doc.doc_id == "d1"


def test_chunk_document_ids_and_sections():
    doc = ingest_markdown("d1", "# Setup\n" + "word " * 300, default_title="Doc")
    chunks = chunk_document(doc, max_chars=200, overlap=40)
    assert len(chunks) > 1
    assert chunks[0].chunk_id == "d1#0"
    assert chunks[1].chunk_id == "d1#1"
    assert all(c.section == "Setup" for c in chunks)
    assert all(c.doc_id == "d1" for c in chunks)


def test_ingest_directory(tmp_path):
    (tmp_path / "a.md").write_text("---\ntitle: Doc A\n---\n# S\nhello")
    (tmp_path / "b.txt").write_text("plain text doc")
    (tmp_path / "skip.py").write_text("print('ignored')")
    docs = ingest_directory(tmp_path)
    assert {d.doc_id for d in docs} == {"a", "b"}
    assert docs[0].title == "Doc A"


def test_ingest_directory_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_directory(tmp_path / "nope")
