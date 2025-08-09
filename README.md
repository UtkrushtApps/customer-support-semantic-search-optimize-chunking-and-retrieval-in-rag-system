# Customer Support Semantic Retrieval - Candidate Task

## Task Overview
You're improving a customer support RAG system that stores 8,000 pre-loaded support docs in a Chroma vector DB. The current chunking strategy (2000 tokens, no overlap, no metadata) leads to poor-quality retrieval. Your job is to:
- Re-implement document chunking (200 tokens per chunk, 40-token overlap).
- Attach chunk-level metadata: support category, priority, creation date (these are given in the input files).
- Insert all chunks with embeddings and metadata into the Chroma database.
- Implement top-5 most relevant document retrieval via cosine similarity.
- **Do not modify API or infrastructure; focus only on chunking, embedding, metadata, and retrieval logic.**

## Guidance
- The database setup, embedding model, and FastAPI endpoints are already in place.
- Refactor `init_vector_db.py` to process all documents using the correct chunking and metadata strategy.
- Complete the placeholder in `rag_retrieval.py` to implement efficient top-k retrieval by cosine similarity, using the `vector_db_client.py` interface.
- Common improvement areas: chunk overlap, token budgeting, attaching structured metadata, ensuring consistent embedding dimensions.
- Check and document improvements in support recall and query relevance.

## Database Access
- **Vector database:** Chroma
- **Host:** `<DROPLET_IP>`
- **Port:** 8000
- **Collection/Index:** support_docs
- **Vector size:** 384 (all-MiniLM-L6-v2)
- **Metadata fields:**
  - `chunk_id` (string) – unique per chunk
  - `doc_id` (string) – original document ID
  - `chunk_index` (int)
  - `category` (string)
  - `priority` (string)
  - `date` (string, yyyy-mm-dd)
  - `start_position` (char offset)
  - `token_count` (int)
- Chunked and embedded docs can be explored using Chroma Python SDK or vector DB admin tools. Confirm counts and metadata via queries.

## Objectives
- Implement correct chunking with overlap and metadata for all documents.
- Attach embeddings using provided sentence-transformer model.
- Store all chunk embeddings and metadata in Chroma.
- Implement top-5 retrieval using cosine similarity.
- Evaluate by running queries from `sample_queries.txt` and measuring recall@k and precision@k.

## How to Verify
- Use API or CLI to submit support queries and inspect returned chunk content and metadata.
- Perform manual spot checks for relevant answers and verify recall@5 correctness.
- Confirm chunk/metadata counts in Chroma to match expectations after re-ingestion.
