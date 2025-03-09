import torch
from tqdm import tqdm
import numpy as np
from util import split_text, split_text_semantically
import json
from typing import List, Dict



def generate_embeddings_with_progress(documents, embedding_model, field="content", batch_size=32):
    from tqdm import tqdm

    embeddings = []
    texts = [doc[field] for doc in documents]
    for start in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[start:start+batch_size]
        batch_embeddings = embedding_model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings.extend(batch_embeddings)

    return np.array(embeddings)

def generate_embeddings_long_text(documents, embedding_model, field="content", chunk_size=512, overlap=50, batch_size=32):

    embeddings = []

    for doc in tqdm(documents, desc="Processing Documents"):
        text = doc[field]
        chunks = split_text(text, chunk_size, overlap)

        if not chunks:
            # print(f"Warning: Document {doc['id']} has no valid chunks.")
            embeddings.append(np.zeros(embedding_model.get_sentence_embedding_dimension()))
            continue

        chunk_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            try:
                batch_embeddings = embedding_model.encode(batch_chunks, convert_to_numpy=True, batch_size=batch_size)
                chunk_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error encoding batch for document {doc['id']} - {e}")

        if chunk_embeddings:
            embeddings.append(np.mean(chunk_embeddings, axis=0))
        else:
            embeddings.append(np.zeros(embedding_model.get_sentence_embedding_dimension()))

    print(f"Generated {len(embeddings)} document embeddings.")
    return np.array(embeddings)

def generate_aggregated_embedding(text, embedding_model,max_tokens=512, method='average'):
    """
    Segment very long text and generate aggregate vectors.
    """

    chunks = split_text_semantically(text, max_tokens=max_tokens)
    embeddings = [embedding_model.encode(chunk) for chunk in chunks]

    if method == 'average':
        aggregated_embedding = np.mean(embeddings, axis=0)
    elif method == 'max_pooling':
        aggregated_embedding = np.max(embeddings, axis=0)
    else:
        raise ValueError("Unsupported aggregation method.")

    return aggregated_embedding
def embedding_documents(documents, include_section=True,max_tokens=512, aggregation_method='average') -> List[Dict]:
    processed_docs = []

    for doc in tqdm(documents.keys(), desc="Processing Documents"):
        doc = documents[doc]
        if include_section:
            sections = doc['sections']
            for section in sections:
                text = section["text"]
                # 调用 generate_aggregated_embedding
                aggregated_embedding = generate_aggregated_embedding(
                    text, max_tokens=max_tokens, method=aggregation_method
                )
                doc_data = {
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "sec_id": section["id"],
                    "text": section["text"],
                    "embedding": aggregated_embedding,
                }
                processed_docs.append(doc_data)
        else:
            doc_data = {
                "doc_id": doc["id"],
                "title": doc["title"],
                "content": doc["text"]
            }
            processed_docs.append(doc_data)
    return processed_docs

def load_processed_embeddings(file_path):

    with open(file_path, "r") as f:
        data = json.load(f)
    for item in data:
        item["embedding"] = np.array(item["embedding"])

    return data

def fix_emmbedding(processed_results):
    valid_results = []
    for i, item in enumerate(processed_results):
        embedding = item["embedding"]
        if isinstance(embedding, (list, np.ndarray)):
            valid_results.append(item)

    expected_dim = 768
    valid_embeddings = []
    metadata = []

    for i, item in enumerate(valid_results):
        embedding = np.array(item["embedding"], dtype="float32")
        if embedding.ndim == 0 or embedding.shape == ():
            continue
        if embedding.shape[0] == expected_dim:
            valid_embeddings.append(embedding)
            metadata.append({
                "doc_id": item["doc_id"],
                "sec_id": item["sec_id"],
                "text": item["text"]
            })

    embeddings = np.array(valid_embeddings, dtype="float32")
    return embeddings, metadata