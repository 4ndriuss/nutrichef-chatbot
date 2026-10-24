# Hybrid retrieval keeps ChromaDB alongside BM25

On this Knowledge base, BM25 alone ranked Excerpts as well as or better than ChromaDB on 5 of 7 test queries, mostly because Chroma's default English-only MiniLM embedding handles Indonesian poorly. We still keep ChromaDB and fuse both engines' rankings with reciprocal rank fusion instead of dropping it, because the project is specified and documented as hybrid RAG with a vector store, and RRF keeps BM25's top hits while letting Chroma add semantically related Excerpts it finds better on some queries. When ChromaDB is unavailable, retrieval runs on BM25 alone.

Fusion takes only the top 5 candidates from each engine. With 10 candidates out of ~36 Excerpts, weak matches that happened to appear in both lists outvoted BM25's correct #1 (relevant Excerpt at #1 dropped from 5/7 to 3/7 queries); with 5, hybrid kept every BM25 #1 and found one more relevant Excerpt than BM25 alone (6/8 vs 5/8 in the top 3). This was measured on 7 hand-judged queries, so revisit it if the Knowledge base grows substantially.

## Considered Options

- **Drop ChromaDB (BM25 only):** ~340 MB fewer dependencies and instant startup, but removes the vector store the project promises.
- **Fallback (Chroma if it builds, else BM25):** the original behaviour; on this data it usually selects the weaker engine.
- **Swap in a multilingual embedding model:** likely the real fix for Chroma's quality; deferred because it changes dependencies and download size.
