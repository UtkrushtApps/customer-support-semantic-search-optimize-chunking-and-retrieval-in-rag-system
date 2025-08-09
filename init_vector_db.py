import os
from sentence_transformers import SentenceTransformer
from config import *
from vector_db_client import get_collection
import re

CHUNK_SIZE = 200
CHUNK_OVERLAP = 40
EMBED_DIM = 384
BATCH_SIZE = 16


def parse_metadata(section):
    headers = {}
    for line in section.strip().splitlines()[:4]:
        if line.startswith('DOC_ID:'):
            headers['doc_id'] = line.replace('DOC_ID:', '').strip()
        elif line.startswith('CATEGORY:'):
            headers['category'] = line.replace('CATEGORY:', '').strip()
        elif line.startswith('PRIORITY:'):
            headers['priority'] = line.replace('PRIORITY:', '').strip()
        elif line.startswith('DATE:'):
            headers['date'] = line.replace('DATE:', '').strip()
    return headers

def tokenize_text(text):
    return re.findall(r'\w+|[.,;!?\n]', text)

def chunk_document(doc_text, chunk_size=200, overlap=40):
    tokens = tokenize_text(doc_text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i+chunk_size]
        chunk_str = ' '.join(chunk_tokens)
        char_pos = len(' '.join(tokens[:i]))
        chunks.append((chunk_str, char_pos, i // chunk_size))
        i += chunk_size - overlap
    return chunks

def main():
    print("[INFO] Loading support docs from data/full_doc.txt ...")
    assert os.path.exists('data/full_doc.txt'), "Support doc file missing."
    with open('data/full_doc.txt', encoding='utf-8') as f:
        full_text = f.read()
    docs = full_text.split('\n\n---\n\n')
    model = SentenceTransformer(EMBED_MODEL_NAME)
    metadatas, chunk_texts, chunk_ids, embeddings = [], [], [], []
    chunk_counter = 0
    print(f"[INFO] {len(docs)} docs to process.")
    for i, section in enumerate(docs):
        headers = parse_metadata(section)
        if not headers.get('doc_id'): continue
        body = section.split('\n', 4)[-1].strip()
        chunks = chunk_document(body, CHUNK_SIZE, CHUNK_OVERLAP)
        chunk_bodies = [c[0] for c in chunks]
        positions = [c[1] for c in chunks]
        indices = [c[2] for c in chunks]
        chunk_ids += [f"{headers['doc_id']}_c{ix}" for ix in indices]
        metadatas += [{
            "chunk_id": f"{headers['doc_id']}_c{ix}",
            "doc_id": headers['doc_id'],
            "chunk_index": ix,
            "category": headers['category'],
            "priority": headers['priority'],
            "date": headers['date'],
            "start_position": pos,
            "token_count": len(chunk_bodies[ix].split()),
        } for ix, pos in zip(indices, positions)]
        chunk_texts += chunk_bodies
    print(f"[INFO] Created {len(chunk_texts)} chunks.")
    for i in range(0, len(chunk_texts), BATCH_SIZE):
        batch = chunk_texts[i:i+BATCH_SIZE]
        emb = model.encode(batch, show_progress_bar=False)
        embeddings += list(emb)
    print(f"[INFO] {len(embeddings)} embeddings generated.")
    collection = get_collection()
    print(f"[INFO] Inserting {len(chunk_texts)} chunks into Chroma ...")
    collection.upsert(ids=chunk_ids, embeddings=embeddings, metadatas=metadatas, documents=chunk_texts)
    print("[COMPLETE] All chunks and metadata inserted.")

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(f"[ERROR] {ex}")
        exit(1)
