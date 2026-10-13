"""
rag_pipeline.py - Retrieval-Augmented Generation (RAG) Engine untuk NutriChef AI
Mendukung Vector Store ChromaDB dengan Embedding serta Fallback BM25/Cosine Similarity.
"""

import os
import re
import math
from typing import List, Dict, Any, Tuple
from collections import Counter
import pypdf

# Directory path default untuk knowledge base
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "data", "knowledge_base")


class DocumentChunk:
    def __init__(self, content: str, source: str, title: str = ""):
        self.content = content.strip()
        self.source = source
        self.title = title

    def to_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "source": self.source,
            "content": self.content
        }


def chunk_markdown_text(text: str, source_name: str) -> List[DocumentChunk]:
    """
    Memecah teks Markdown berdasarkan header (## atau ###) agar setiap babak/resep
    menjadi satu chunk utuh yang mempertahankan konteks semantiknya.
    """
    chunks: List[DocumentChunk] = []
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
        
        # Jika chunk sangat panjang (> 1800 karakter), pecah lagi dengan paragraf
        if len(clean_section) > 1800:
            sub_paragraphs = clean_section.split('\n\n')
            current_sub = ""
            for p in sub_paragraphs:
                if len(current_sub) + len(p) < 1500:
                    current_sub += p + "\n\n"
                else:
                    if current_sub.strip():
                        chunks.append(DocumentChunk(current_sub.strip(), source_name, title))
                    current_sub = p + "\n\n"
            if current_sub.strip():
                chunks.append(DocumentChunk(current_sub.strip(), source_name, title))
        else:
            chunks.append(DocumentChunk(clean_section, source_name, title))
            
    return chunks


def extract_text_from_pdf(file_bytes) -> str:
    """Mengekstrak teks dari file PDF yang diunggah pengguna."""
    reader = pypdf.PdfReader(file_bytes)
    extracted = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            extracted.append(f"--- Halaman {i+1} ---\n{page_text}")
    return "\n\n".join(extracted)


class BM25Retriever:
    """
    Engine pencarian berbasis BM25 & Term-Frequency Cosine Similarity.
    Sangat andal, deterministik, instan, tanpa latensi jaringan atau dependensi GPU.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[DocumentChunk] = []
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = Counter()
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        # Tokenizer sederhana: lowercase, buang tanda baca
        tokens = re.findall(r'\b[a-zA-Z0-9_\-]+\b', text.lower())
        # Filter stop words singkat bahasa indonesia & inggris
        stopwords = {
            'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 'dia', 'dua',
            'ia', 'seperti', 'jika', 'sehingga', 'kembali', 'dan', 'ini', 'karena', 'oleh',
            'saat', 'harus', 'itu', 'adalah', 'the', 'and', 'is', 'in', 'of', 'to', 'with', 'for'
        }
        return [t for t in tokens if len(t) > 2 and t not in stopwords]

    def fit(self, chunks: List[DocumentChunk]):
        self.corpus = chunks
        self.doc_len = []
        self.doc_freqs = Counter()
        
        tokenized_corpus = []
        for chunk in chunks:
            tokens = self._tokenize(chunk.title + " " + chunk.content)
            tokenized_corpus.append(tokens)
            self.doc_len.append(len(tokens))
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.doc_freqs[t] += 1
                
        n_docs = len(chunks)
        self.avg_doc_len = sum(self.doc_len) / max(n_docs, 1)
        
        # Hitung IDF
        self.idf = {}
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        if not self.corpus:
            return []
            
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return [(self.corpus[i], 0.1) for i in range(min(top_k, len(self.corpus)))]
            
        scores = []
        for idx, chunk in enumerate(self.corpus):
            chunk_tokens = self._tokenize(chunk.title + " " + chunk.content)
            doc_len = self.doc_len[idx]
            token_counts = Counter(chunk_tokens)
            
            score = 0.0
            for qt in q_tokens:
                if qt in token_counts:
                    tf = token_counts[qt]
                    idf_val = self.idf.get(qt, 0.1)
                    # Rumus BM25 scoring
                    denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1)))
                    score += idf_val * (tf * (self.k1 + 1)) / max(denom, 1e-6)
                    
            # Boost jika kata kunci ada di judul
            for qt in q_tokens:
                if qt in chunk.title.lower():
                    score += 1.5
                    
            scores.append((chunk, score))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class RAGPipeline:
    """
    Pipeline Utama RAG untuk NutriChef AI:
    - Membaca basis pengetahuan lokal (Markdown resep, gizi, diet)
    - Menerima file upload (PDF/TXT)
    - Menyimpan ke dalam vector store (ChromaDB jika tersedia, atau BM25 Hybrid)
    """
    def __init__(self, data_dir: str = KNOWLEDGE_BASE_DIR):
        self.data_dir = data_dir
        self.chunks: List[DocumentChunk] = []
        self.chroma_collection = None
        self.embedding_fn = None
        self.bm25_retriever = BM25Retriever()
        self.engine_type = "BM25 Hybrid Engine"
        
        self._init_engine()

    def _init_engine(self):
        # Coba inisialisasi ChromaDB jika pustaka tersedia
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            # Buat client ephemeral in-memory chromadb
            client = chromadb.Client()
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            self.chroma_collection = client.create_collection(
                name="nutrichef_kb",
                embedding_function=self.embedding_fn
            )
            self.engine_type = "ChromaDB Vector Store"
        except Exception:
            # Fallback ke BM25 Retriever
            self.chroma_collection = None
            self.engine_type = "BM25 Hybrid Engine"

    def load_knowledge_base(self):
        """Membaca semua file panduan & resep di folder data/knowledge_base."""
        self.chunks = []
        if os.path.exists(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if filename.endswith(".md") or filename.endswith(".txt"):
                    filepath = os.path.join(self.data_dir, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                        file_chunks = chunk_markdown_text(content, source_name=filename)
                        self.chunks.extend(file_chunks)
                    except Exception as e:
                        print(f"Gagal membaca {filename}: {e}")

        # Update index
        self._reindex()

    def add_uploaded_file(self, filename: str, file_bytes):
        """Memproses file PDF atau teks yang diunggah pengguna."""
        content = ""
        if filename.lower().endswith(".pdf"):
            content = extract_text_from_pdf(file_bytes)
        else:
            content = file_bytes.read().decode("utf-8", errors="ignore")
            
        if content.strip():
            new_chunks = chunk_markdown_text(content, source_name=f"Upload: {filename}")
            self.chunks.extend(new_chunks)
            self._reindex()
            return len(new_chunks)
        return 0

    def _reindex(self):
        """Sinkronisasi data ke BM25 dan ChromaDB (jika aktif)."""
        # Fit BM25
        self.bm25_retriever.fit(self.chunks)
        
        # Index ke ChromaDB jika aktif
        if self.chroma_collection is not None and self.chunks:
            try:
                # Reset collection
                ids = [f"chunk_{i}" for i in range(len(self.chunks))]
                documents = [c.content for c in self.chunks]
                metadatas = [{"title": c.title, "source": c.source} for c in self.chunks]
                
                # Delete existing documents if any
                existing = self.chroma_collection.get()
                if existing and existing.get("ids"):
                    self.chroma_collection.delete(ids=existing["ids"])
                    
                self.chroma_collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            except Exception as e:
                # Fallback ke BM25 jika terjadi kendala pada chromadb saat runtime
                print(f"ChromaDB index warning, falling back to BM25: {e}")

    def retrieve(self, query: str, top_k: int = 3) -> List[DocumentChunk]:
        """
        Mengambil k chunk paling relevan untuk pertanyaan pengguna.
        """
        if not self.chunks:
            return []
            
        # Jika ChromaDB aktif dan siap
        if self.chroma_collection is not None:
            try:
                results = self.chroma_collection.query(
                    query_texts=[query],
                    n_results=min(top_k, len(self.chunks))
                )
                if results and results.get("documents") and results["documents"][0]:
                    retrieved_chunks = []
                    docs = results["documents"][0]
                    metas = results["metadatas"][0] if results.get("metadatas") else []
                    for i, doc in enumerate(docs):
                        meta = metas[i] if i < len(metas) else {}
                        retrieved_chunks.append(
                            DocumentChunk(
                                content=doc,
                                source=meta.get("source", "Knowledge Base"),
                                title=meta.get("title", "")
                            )
                        )
                    return retrieved_chunks
            except Exception as e:
                print(f"Chroma query failed, fallback to BM25: {e}")

        # Fallback ke BM25
        bm25_results = self.bm25_retriever.search(query, top_k=top_k)
        return [chunk for chunk, score in bm25_results]

    def format_context_for_prompt(self, query: str, top_k: int = 3) -> str:
        """Mengonversi hasil retrieve menjadi string konteks yang siap disisipkan ke LLM."""
        matched_chunks = self.retrieve(query, top_k=top_k)
        if not matched_chunks:
            return ""
            
        formatted = []
        for i, chunk in enumerate(matched_chunks, 1):
            formatted.append(f"[Konteks #{i} | Sumber: {chunk.source} | Topik: {chunk.title}]\n{chunk.content}")
            
        return "\n\n".join(formatted)
