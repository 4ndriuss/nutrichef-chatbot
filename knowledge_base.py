"""
knowledge_base.py - Knowledge base NutriChef AI: Document, Excerpt, dan retrieval hybrid (BM25 + ChromaDB)
"""

import math
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import pypdf

# Directory path default untuk Document kurasi
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "data", "knowledge_base")
# Jumlah Excerpt yang dijadikan Sources per pertanyaan
SOURCES_TOP_K = 3
# Jumlah kandidat teratas dari tiap engine yang ikut digabung. Sengaja kecil: dengan Knowledge base
# sekecil ini, kandidat yang banyak membuat kecocokan lemah di kedua engine mengalahkan peringkat #1
CANDIDATES_PER_ENGINE = 5
# Konstanta standar Reciprocal Rank Fusion (RRF)
RRF_K = 60

# Filter stop words singkat bahasa indonesia & inggris
STOPWORDS = {
    'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua',
    'ia', 'seperti', 'jika', 'sehingga', 'kembali', 'dan', 'ini', 'karena', 'oleh',
    'saat', 'harus', 'itu', 'adalah', 'the', 'and', 'is', 'in', 'of', 'to', 'with', 'for'
}


@dataclass(frozen=True)
class Excerpt:
    """Satu bagian berjudul dari sebuah Document; unit yang diambil dari Knowledge base."""
    content: str
    source: str
    title: str = ""


def split_into_excerpts(text: str, source_name: str) -> List[Excerpt]:
    """
    Memecah teks Markdown berdasarkan header (## atau ###) agar setiap babak/resep
    menjadi satu Excerpt utuh yang mempertahankan konteks semantiknya.
    """
    excerpts: List[Excerpt] = []
    # Split by level 2 or level 3 markdown headers
    sections = re.split(r'\n(?=##+\s+)', text)

    for section in sections:
        clean_section = section.strip()
        if not clean_section or len(clean_section) < 30:
            continue

        # Ekstrak judul dari baris pertama jika ada
        lines = clean_section.split('\n')
        first_line = lines[0].strip()
        title = first_line.lstrip('#').strip() if first_line.startswith('#') else source_name

        # Jika bagian sangat panjang (> 1800 karakter), pecah lagi dengan paragraf
        if len(clean_section) > 1800:
            sub_paragraphs = clean_section.split('\n\n')
            current_sub = ""
            for p in sub_paragraphs:
                if len(current_sub) + len(p) < 1500:
                    current_sub += p + "\n\n"
                else:
                    if current_sub.strip():
                        excerpts.append(Excerpt(current_sub.strip(), source_name, title))
                    current_sub = p + "\n\n"
            if current_sub.strip():
                excerpts.append(Excerpt(current_sub.strip(), source_name, title))
        else:
            excerpts.append(Excerpt(clean_section, source_name, title))

    return excerpts


def extract_text_from_pdf(file) -> str:
    """Mengekstrak teks dari file PDF yang diunggah pengguna."""
    reader = pypdf.PdfReader(file)
    extracted = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            extracted.append(f"--- Halaman {i+1} ---\n{page_text}")
    return "\n\n".join(extracted)


def _tokenize(text: str) -> List[str]:
    # Tokenizer sederhana: lowercase, buang tanda baca
    tokens = re.findall(r'\b[a-zA-Z0-9_\-]+\b', text.lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


class BM25Engine:
    """
    Engine pencarian leksikal berbasis BM25.
    Deterministik, instan, tanpa latensi jaringan atau dependensi GPU.
    Setiap Excerpt di-tokenisasi sekali saat ditambahkan.
    """
    name = "BM25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._excerpts: List[Excerpt] = []
        self._term_counts: List[Counter] = []
        self._doc_len: List[int] = []
        self._doc_freqs: Counter = Counter()
        self._idf: Dict[str, float] = {}
        self._avg_doc_len = 0.0

    def add(self, excerpts: List[Excerpt]):
        for excerpt in excerpts:
            term_counts = Counter(_tokenize(excerpt.title + " " + excerpt.content))
            self._excerpts.append(excerpt)
            self._term_counts.append(term_counts)
            self._doc_len.append(sum(term_counts.values()))
            for token in term_counts:
                self._doc_freqs[token] += 1

        n_docs = len(self._excerpts)
        self._avg_doc_len = sum(self._doc_len) / max(n_docs, 1)
        self._idf = {
            token: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in self._doc_freqs.items()
        }

    def search(self, question: str, k: int) -> List[Excerpt]:
        """Mengembalikan maksimal k Excerpt dengan skor > 0, terurut dari yang paling relevan."""
        q_tokens = _tokenize(question)
        scored = []
        for excerpt, term_counts, doc_len in zip(self._excerpts, self._term_counts, self._doc_len):
            score = 0.0
            for qt in q_tokens:
                tf = term_counts.get(qt, 0)
                if tf:
                    # Rumus BM25 scoring
                    denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(self._avg_doc_len, 1)))
                    score += self._idf.get(qt, 0.1) * (tf * (self.k1 + 1)) / max(denom, 1e-6)

            # Boost jika kata kunci ada di judul
            for qt in q_tokens:
                if qt in excerpt.title.lower():
                    score += 1.5

            if score > 0:
                scored.append((excerpt, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [excerpt for excerpt, _ in scored[:k]]


class ChromaEngine:
    """
    Engine semantik berbasis embedding ChromaDB (in-memory, model all-MiniLM-L6-v2).
    Model embedding diunduh saat Excerpt pertama ditambahkan; jika gagal, Knowledge base melepas engine ini.
    """
    name = "ChromaDB"

    def __init__(self):
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.Client()
        self._collection = client.create_collection(
            name=f"nutrichef_kb_{uuid.uuid4().hex}",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        self._excerpts: List[Excerpt] = []

    def add(self, excerpts: List[Excerpt]):
        if not excerpts:
            return
        # Hanya Excerpt baru yang di-embed; id stabil mengikuti urutan penambahan
        start = len(self._excerpts)
        self._collection.add(
            ids=[f"excerpt_{start + i}" for i in range(len(excerpts))],
            documents=[f"{e.title}\n{e.content}" for e in excerpts]
        )
        self._excerpts.extend(excerpts)

    def search(self, question: str, k: int) -> List[Excerpt]:
        if not self._excerpts:
            return []
        results = self._collection.query(query_texts=[question], n_results=min(k, len(self._excerpts)))
        return [self._excerpts[int(excerpt_id.split("_")[1])] for excerpt_id in results["ids"][0]]


def default_engines() -> list:
    """BM25 selalu tersedia; ChromaDB ikut jika pustakanya terpasang."""
    engines = [BM25Engine()]
    try:
        engines.append(ChromaEngine())
    except Exception as e:
        print(f"ChromaDB tidak tersedia, memakai BM25 saja: {e}")
    return engines


def reciprocal_rank_fusion(rankings: List[List[Excerpt]]) -> List[Excerpt]:
    """
    Menggabungkan peringkat beberapa engine: skor tiap Excerpt = Σ 1/(RRF_K + rank).
    Skor seri mengikuti urutan engine pertama (urutan kemunculan dipertahankan oleh sort yang stabil).
    """
    scores: Dict[Excerpt, float] = {}
    for ranking in rankings:
        for rank, excerpt in enumerate(ranking, 1):
            scores[excerpt] = scores.get(excerpt, 0.0) + 1 / (RRF_K + rank)
    return sorted(scores, key=lambda excerpt: scores[excerpt], reverse=True)


class KnowledgeBase:
    def __init__(self, directory: str = KNOWLEDGE_BASE_DIR, engines=None):
        self._excerpts: List[Excerpt] = []
        self._engines = list(engines) if engines is not None else default_engines()
        self._add_excerpts(self._load_directory(directory))

    @property
    def engine_name(self) -> str:
        names = [engine.name for engine in self._engines]
        if len(names) > 1:
            return f"Hybrid ({' + '.join(names)})"
        return names[0] if names else "Tidak ada engine"

    @property
    def excerpt_count(self) -> int:
        return len(self._excerpts)

    def find_sources(self, question: str) -> str:
        """Mengambil Excerpt paling relevan dan memformatnya sebagai Sources untuk system prompt."""
        rankings = []
        for engine in list(self._engines):
            try:
                rankings.append(engine.search(question, CANDIDATES_PER_ENGINE))
            except Exception as e:
                self._drop_engine(engine, e)
        matched = reciprocal_rank_fusion(rankings)[:SOURCES_TOP_K]
        formatted = []
        for i, excerpt in enumerate(matched, 1):
            formatted.append(f"[Konteks #{i} | Sumber: {excerpt.source} | Topik: {excerpt.title}]\n{excerpt.content}")
        return "\n\n".join(formatted)

    def add_document(self, filename: str, file) -> int:
        """Memproses Document PDF atau teks yang diunggah pengguna; mengembalikan jumlah Excerpt baru."""
        if filename.lower().endswith(".pdf"):
            content = extract_text_from_pdf(file)
        else:
            content = file.read().decode("utf-8", errors="ignore")

        if not content.strip():
            return 0
        excerpts = split_into_excerpts(content, source_name=f"Upload: {filename}")
        self._add_excerpts(excerpts)
        return len(excerpts)

    def _load_directory(self, directory) -> List[Excerpt]:
        """Membaca semua Document kurasi (.md/.txt) di direktori Knowledge base."""
        excerpts: List[Excerpt] = []
        if os.path.exists(directory):
            for filename in sorted(os.listdir(directory)):
                if filename.endswith(".md") or filename.endswith(".txt"):
                    filepath = os.path.join(directory, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                        excerpts.extend(split_into_excerpts(content, source_name=filename))
                    except Exception as e:
                        print(f"Gagal membaca {filename}: {e}")
        return excerpts

    def _add_excerpts(self, excerpts: List[Excerpt]):
        self._excerpts.extend(excerpts)
        for engine in list(self._engines):
            try:
                engine.add(excerpts)
            except Exception as e:
                self._drop_engine(engine, e)

    def _drop_engine(self, engine, error: Exception):
        """Engine yang gagal dilepas permanen agar tidak dicoba ulang di setiap pertanyaan."""
        self._engines.remove(engine)
        print(f"Engine {engine.name} gagal dan dinonaktifkan, lanjut dengan {self.engine_name}: {error}")
