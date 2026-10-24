"""
test_knowledge_base.py - Pengujian Knowledge base lewat interface-nya (Document dari direktori sementara)
"""

import io
import re
import sys
from pathlib import Path

import pytest

from knowledge_base import BM25Engine, KnowledgeBase

RESEP = """# Resep Sehat

## Pepes Ayam Kemangi
Dada ayam dibumbui kemangi lalu dikukus dalam daun pisang. Sekitar 195 kkal per porsi.

## Sup Tahu Sayur
Tahu putih, wortel, dan sawi direbus dalam kaldu bening. Rendah kalori dan tinggi serat.
"""

DIET = """# Panduan Diet

## Diet Rendah Gula
Pilih bahan dengan indeks glikemik di bawah 55 seperti oatmeal dan beras merah.
"""


def sections(names):
    """Document Markdown dengan satu Excerpt berjudul untuk setiap nama."""
    return "\n\n".join(
        f"## {name}\nBagian {name} berisi teks contoh yang cukup panjang untuk menjadi Excerpt." for name in names
    )


ABCD = sections(["Alpha", "Bravo", "Charlie", "Delta"])


class FakeEngine:
    """Engine palsu: mengurutkan Excerpt sesuai daftar judul yang ditentukan dan mencatat pencarian."""

    def __init__(self, name, ranking=(), fail_on=None):
        self.name = name
        self.ranking = list(ranking)
        self.fail_on = fail_on
        self.excerpts = []
        self.searches = 0

    def add(self, excerpts):
        if self.fail_on == "add":
            raise RuntimeError("engine gagal saat indexing")
        self.excerpts.extend(excerpts)

    def search(self, question, k):
        self.searches += 1
        if self.fail_on == "search":
            raise RuntimeError("engine gagal saat pencarian")
        return [e for title in self.ranking for e in self.excerpts if e.title == title][:k]


def topics(sources):
    return re.findall(r"Topik: ([^\]]+)\]", sources)


def write_documents(directory):
    (directory / "resep.md").write_text(RESEP, encoding="utf-8")
    (directory / "diet.md").write_text(DIET, encoding="utf-8")
    (directory / "catatan.json").write_text('{"bukan": "dokumen"}', encoding="utf-8")


def test_building_loads_every_excerpt_of_the_curated_documents(tmp_path):
    write_documents(tmp_path)

    knowledge_base = KnowledgeBase(directory=tmp_path, engines=[BM25Engine()])

    assert knowledge_base.excerpt_count == 3


def test_find_sources_returns_only_matching_excerpts_formatted_for_the_prompt(tmp_path):
    write_documents(tmp_path)
    knowledge_base = KnowledgeBase(directory=tmp_path, engines=[BM25Engine()])

    sources = knowledge_base.find_sources("pepes ayam kemangi")

    assert sources == (
        "[Konteks #1 | Sumber: resep.md | Topik: Pepes Ayam Kemangi]\n"
        "## Pepes Ayam Kemangi\n"
        "Dada ayam dibumbui kemangi lalu dikukus dalam daun pisang. Sekitar 195 kkal per porsi."
    )


def test_find_sources_is_empty_when_nothing_matches(tmp_path):
    write_documents(tmp_path)
    knowledge_base = KnowledgeBase(directory=tmp_path, engines=[BM25Engine()])

    assert knowledge_base.find_sources("xylofon kuantum") == ""


def test_find_sources_fuses_engine_rankings_by_reciprocal_rank(tmp_path):
    (tmp_path / "abcd.md").write_text(ABCD, encoding="utf-8")
    lexical = FakeEngine("Lexical", ranking=["Alpha", "Bravo", "Charlie"])
    semantic = FakeEngine("Semantic", ranking=["Delta", "Bravo", "Alpha"])
    knowledge_base = KnowledgeBase(directory=tmp_path, engines=[lexical, semantic])

    assert topics(knowledge_base.find_sources("apa saja")) == ["Alpha", "Bravo", "Delta"]


def test_fusion_ties_follow_the_first_engines_order(tmp_path):
    (tmp_path / "abcd.md").write_text(ABCD, encoding="utf-8")
    lexical = FakeEngine("Lexical", ranking=["Charlie"])
    semantic = FakeEngine("Semantic", ranking=["Bravo"])
    knowledge_base = KnowledgeBase(directory=tmp_path, engines=[lexical, semantic])

    assert topics(knowledge_base.find_sources("apa saja")) == ["Charlie", "Bravo"]


def test_engine_name_names_every_engine_in_use(tmp_path):
    (tmp_path / "abcd.md").write_text(ABCD, encoding="utf-8")

    hybrid = KnowledgeBase(directory=tmp_path, engines=[FakeEngine("BM25"), FakeEngine("ChromaDB")])
    single = KnowledgeBase(directory=tmp_path, engines=[FakeEngine("BM25")])

    assert hybrid.engine_name == "Hybrid (BM25 + ChromaDB)"
    assert single.engine_name == "BM25"


def test_engine_failing_while_indexing_is_dropped(tmp_path):
    (tmp_path / "abcd.md").write_text(ABCD, encoding="utf-8")
    broken = FakeEngine("ChromaDB", ranking=["Delta"], fail_on="add")

    knowledge_base = KnowledgeBase(directory=tmp_path, engines=[FakeEngine("BM25", ranking=["Alpha"]), broken])

    assert knowledge_base.engine_name == "BM25"
    assert topics(knowledge_base.find_sources("apa saja")) == ["Alpha"]
    assert broken.searches == 0


def test_engine_failing_at_search_is_dropped_for_good(tmp_path):
    (tmp_path / "abcd.md").write_text(ABCD, encoding="utf-8")
    broken = FakeEngine("ChromaDB", ranking=["Delta"], fail_on="search")
    knowledge_base = KnowledgeBase(directory=tmp_path, engines=[FakeEngine("BM25", ranking=["Alpha"]), broken])

    first = knowledge_base.find_sources("apa saja")
    second = knowledge_base.find_sources("apa lagi")

    assert topics(first) == topics(second) == ["Alpha"]
    assert knowledge_base.engine_name == "BM25"
    assert broken.searches == 1


def make_pdf(text):
    """Membuat PDF satu halaman yang valid berisi satu baris teks."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    pdf += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return io.BytesIO(pdf)


def test_uploaded_text_document_becomes_findable(tmp_path):
    write_documents(tmp_path)
    knowledge_base = KnowledgeBase(directory=tmp_path, engines=[BM25Engine()])
    upload = io.BytesIO("## Tempe Bacem\nTempe direbus dengan gula aren dan bumbu, lalu digoreng sebentar.".encode())

    added = knowledge_base.add_document("catatan.txt", upload)

    assert added == 1
    assert knowledge_base.excerpt_count == 4
    assert knowledge_base.find_sources("tempe bacem").startswith(
        "[Konteks #1 | Sumber: Upload: catatan.txt | Topik: Tempe Bacem]\n"
    )


def test_uploaded_pdf_document_becomes_findable(tmp_path):
    knowledge_base = KnowledgeBase(directory=tmp_path, engines=[BM25Engine()])

    added = knowledge_base.add_document("menu.PDF", make_pdf("Tempe bacem tinggi protein nabati dan rendah lemak jenuh."))

    assert added == 1
    sources = knowledge_base.find_sources("tempe protein")
    assert sources.startswith("[Konteks #1 | Sumber: Upload: menu.PDF | Topik: Upload: menu.PDF]\n")
    assert "Tempe bacem tinggi protein nabati" in sources


def test_uploaded_document_without_text_adds_nothing(tmp_path):
    knowledge_base = KnowledgeBase(directory=tmp_path, engines=[BM25Engine()])

    assert knowledge_base.add_document("kosong.txt", io.BytesIO(b"   \n")) == 0
    assert knowledge_base.excerpt_count == 0


def test_default_engines_fall_back_to_bm25_when_chromadb_is_unavailable(tmp_path, monkeypatch):
    write_documents(tmp_path)
    monkeypatch.setitem(sys.modules, "chromadb", None)  # import chromadb -> ImportError

    knowledge_base = KnowledgeBase(directory=tmp_path)

    assert knowledge_base.engine_name == "BM25"
    assert topics(knowledge_base.find_sources("pepes ayam kemangi")) == ["Pepes Ayam Kemangi"]


CHROMA_MODEL = Path.home() / ".cache" / "chroma" / "onnx_models" / "all-MiniLM-L6-v2" / "onnx" / "model.onnx"


@pytest.mark.skipif(not CHROMA_MODEL.exists(), reason="model embedding ChromaDB belum ter-cache (butuh unduhan)")
def test_chromadb_engine_finds_semantically_matching_excerpts_including_uploads(tmp_path):
    pytest.importorskip("chromadb")
    from knowledge_base import ChromaEngine

    write_documents(tmp_path)
    knowledge_base = KnowledgeBase(directory=tmp_path, engines=[ChromaEngine()])
    knowledge_base.add_document(
        "catatan.txt", io.BytesIO("## Tempe Bacem\nTempe direbus dengan gula aren lalu digoreng sebentar.".encode())
    )

    assert knowledge_base.engine_name == "ChromaDB"
    assert topics(knowledge_base.find_sources("pepes ayam kemangi"))[0] == "Pepes Ayam Kemangi"
    assert topics(knowledge_base.find_sources("tempe bacem"))[0] == "Tempe Bacem"


def test_weak_matches_found_by_both_engines_do_not_outvote_a_top_hit(tmp_path):
    names = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf",
             "Hotel", "India", "Juliet", "Kilo", "Lima"]
    (tmp_path / "nato.md").write_text(sections(names), encoding="utf-8")
    lexical = FakeEngine("Lexical", ranking=["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf"])
    semantic = FakeEngine("Semantic", ranking=["Hotel", "India", "Juliet", "Kilo", "Lima", "Foxtrot", "Golf"])
    knowledge_base = KnowledgeBase(directory=tmp_path, engines=[lexical, semantic])

    # Foxtrot & Golf hanya kecocokan lemah (peringkat 6-7) di kedua engine; tidak boleh mengalahkan peringkat #1
    assert topics(knowledge_base.find_sources("apa saja")) == ["Alpha", "Hotel", "Bravo"]


def test_identical_excerpts_uploaded_twice_are_ranked_separately(tmp_path):
    (tmp_path / "abcd.md").write_text(ABCD, encoding="utf-8")
    lexical = FakeEngine("Lexical", ranking=["Alpha", "Echo"])
    semantic = FakeEngine("Semantic", ranking=["Delta"])
    knowledge_base = KnowledgeBase(directory=tmp_path, engines=[lexical, semantic])
    for _ in range(2):
        knowledge_base.add_document("echo.txt", io.BytesIO(sections(["Echo"]).encode()))

    # Dua salinan Echo (peringkat 2 & 3) tidak boleh digabung menjadi satu suara yang mengalahkan peringkat #1
    assert topics(knowledge_base.find_sources("apa saja")) == ["Alpha", "Delta", "Echo"]


def test_engine_failing_on_a_later_upload_is_dropped_and_the_upload_still_counts(tmp_path):
    (tmp_path / "abcd.md").write_text(ABCD, encoding="utf-8")
    broken = FakeEngine("ChromaDB", ranking=["Echo"])
    knowledge_base = KnowledgeBase(directory=tmp_path, engines=[FakeEngine("BM25", ranking=["Echo"]), broken])
    broken.fail_on = "add"

    added = knowledge_base.add_document("echo.txt", io.BytesIO(sections(["Echo"]).encode()))

    assert added == 1
    assert knowledge_base.excerpt_count == 5
    assert knowledge_base.engine_name == "BM25"
    assert topics(knowledge_base.find_sources("apa saja")) == ["Echo"]


def test_default_engines_put_bm25_first_so_fusion_ties_follow_bm25(tmp_path):
    pytest.importorskip("chromadb")

    # Tanpa Document, ChromaDB tidak perlu mengunduh model embedding
    assert KnowledgeBase(directory=tmp_path).engine_name == "Hybrid (BM25 + ChromaDB)"
