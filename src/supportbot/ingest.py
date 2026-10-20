"""Document ingestion: raw files -> chunked corpus.

Supports Markdown (with optional YAML front-matter and `#` section
headings) and plain text. Files are split into overlapping word-window
chunks that preserve their section heading for citation context.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import Chunk, Document

FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Split a leading `---` YAML-ish block into (metadata, body).

    Only simple `key: value` lines are understood — enough for
    `title:`/`tags:` style headers without pulling in PyYAML.
    """
    match = FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    metadata: dict = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
    return metadata, raw[match.end() :]


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown body into (heading, text) sections.

    Text before the first heading lands in an untitled section so nothing
    is silently dropped.
    """
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []
    for line in body.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = heading.group(2)
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_heading, "\n".join(current_lines).strip()))
    return [(h, t) for h, t in sections if t]


def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> list[str]:
    """Split text into overlapping word-window chunks.

    Words are the unit (never mid-word splits); overlap keeps sentence
    context across chunk boundaries. A single over-long word stream is
    still emitted rather than dropped.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not (0 <= overlap < max_chars):
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start
        size = 0
        while end < len(words) and size + len(words[end]) + 1 <= max_chars:
            size += len(words[end]) + 1
            end += 1
        if end == start:  # one word longer than max_chars: take it anyway
            end = start + 1
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        # Next window starts `overlap` chars back so chunk context repeats.
        kept_words = 0
        kept_chars = 0
        while kept_words < (end - start) and kept_chars < overlap:
            kept_words += 1
            kept_chars += len(words[end - kept_words]) + 1
        start = max(start + 1, end - kept_words)
    return chunks


def ingest_markdown(
    doc_id: str,
    raw: str,
    source: str = "",
    default_title: str = "",
    max_chars: int = 800,
    overlap: int = 120,
) -> Document:
    """Parse one markdown/text file into a Document (unchunked)."""
    metadata, body = parse_front_matter(raw)
    title = metadata.get("title") or default_title or doc_id
    return Document(
        doc_id=doc_id, title=title, source=source, text=body, metadata=metadata
    )


def chunk_document(
    doc: Document, max_chars: int = 800, overlap: int = 120
) -> list[Chunk]:
    """Cut a Document into section-aware Chunks."""
    chunks: list[Chunk] = []
    index = 0
    for section, section_text in split_sections(doc.text):
        for piece in chunk_text(section_text, max_chars=max_chars, overlap=overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{index}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source=doc.source,
                    text=piece,
                    section=section,
                    metadata=dict(doc.metadata),
                )
            )
            index += 1
    # Fallback: documents with no splittable text still yield one chunk so
    # they remain searchable by title metadata.
    if not chunks and doc.text.strip():
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#0",
                doc_id=doc.doc_id,
                title=doc.title,
                source=doc.source,
                text=doc.text.strip()[:max_chars],
                metadata=dict(doc.metadata),
            )
        )
    return chunks


def ingest_directory(
    path: str | Path,
    extensions: tuple[str, ...] = (".md", ".txt"),
    max_chars: int = 800,
    overlap: int = 120,
) -> list[Document]:
    """Ingest every matching file under a directory into Documents."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {path}")
    documents: list[Document] = []
    for file in sorted(root.rglob("*")):
        if not file.is_file() or file.suffix.lower() not in extensions:
            continue
        raw = file.read_text(encoding="utf-8")
        doc_id = file.stem
        default_title = doc_id.replace("-", " ").replace("_", " ").title()
        documents.append(
            ingest_markdown(
                doc_id,
                raw,
                source=str(file),
                default_title=default_title,
                max_chars=max_chars,
                overlap=overlap,
            )
        )
    return documents
