import faiss
import numpy as np
import torch
from sklearn.preprocessing import normalize


def build_faiss_index_normalize(embeddings):
    embeddings = normalize(embeddings, axis=1)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve_documents_with_scores_new(query, index, doc_embeddings, documents, embedding_model, top_k=3):
    query_embedding = embedding_model.encode([query])[0]

    distances, indices = index.search(query_embedding.reshape(1, -1), k=top_k)

    retrieved_docs_with_scores = []
    for i, idx in enumerate(indices[0]):
        doc = documents[idx]
        score = distances[0][i]

        doc_embedding = doc_embeddings[idx]
        cosine_similarity = (
                query_embedding @ doc_embedding /
                (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
        )
        retrieved_docs_with_scores.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "content": doc["content"],
            "score": score,
            "cosine_similarity": cosine_similarity
        })

    retrieved_docs_with_scores.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    return retrieved_docs_with_scores


def retrieve_documents_with_scores_chunk(query, index, doc_embeddings, documents, embedding_model, top_k=3):
    query_embedding = embedding_model.encode([query])[0]

    distances, indices = index.search(query_embedding.reshape(1, -1), k=top_k)

    retrieved_docs_with_scores = []

    for i, idx in enumerate(indices[0]):
        doc = documents[idx]
        score = distances[0][i]
        doc_embedding = doc_embeddings[idx]

        cosine_similarity = (
                query_embedding @ doc_embedding /
                (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
        )
        retrieved_docs_with_scores.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "sec_id": doc["sec_id"],
            "content": doc["text"],
            "score": score,
            "cosine_similarity": cosine_similarity
        })

    retrieved_docs_with_scores.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    return retrieved_docs_with_scores

def extract_answer(question, context, qa_model, tokenizer, max_length=512,expand_factor=5):
    if not question or not context:
        return "Question or context is empty."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qa_model.to(device)

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length"
    ).to(device)

    outputs = qa_model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    if answer_start > answer_end:
        answer_start, answer_end = answer_end, answer_start
    extended_start = max(0, answer_start - expand_factor)
    extended_end = min(inputs.input_ids.size(1), answer_end + expand_factor)

    answer = tokenizer.decode(inputs.input_ids[0][extended_start:extended_end], skip_special_tokens=True)

    return answer.strip()
