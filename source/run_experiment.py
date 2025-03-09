import argparse
import os
import json
import torch
from sentence_transformers import SentenceTransformer
from transformers import logging
import warnings

from llm_api import (
    prompt_from_question_to_keywords, llm_generate, get_model, cfg
)
from util import (
    load_data,
    save_embeddings,
    load_embeddings,
    build_document_index,
    find_documents_by_id_fast,
    save_preprocessed_data,
    preprocess_documents,
    dict_to_str,
    convert_ndarray_to_list
)
from embedding import (
    generate_embeddings_long_text,
    embedding_documents,
    load_processed_embeddings,
    fix_emmbedding
)
from retrieve import (
    build_faiss_index,
    retrieve_documents_with_scores_new,
    retrieve_documents_with_scores_chunk
)
from bm25 import (
    preprocess_documents_bm25,
    build_bm25_index,
    retrieve_documents_bm25,
    save_bm25_state,
    load_bm25_state
)

# Suppress warnings and unnecessary logs
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def initialize_paths(args):
    """Initialize and return essential paths."""
    data_path = args.data_path
    weights_path = args.weights_path
    return data_path, weights_path
    # current_directory = os.getcwd()
    # parent_directory = os.path.dirname(current_directory)
    # data_path = os.path.join(parent_directory, "data")
    # weights_path = os.path.join(data_path, "weights")
    # return data_path, weights_path

def load_or_preprocess_documents(technotes_path, documents_path):
    """Load or preprocess documents."""
    if os.path.exists(documents_path):
        print(f"Find Documents: {documents_path}\nLoading Documents: {documents_path}")
        return load_data(documents_path)
    else:
        documents = preprocess_documents(technotes_path, False)
        print(f"Data preprocessing is complete!\nNumber of Documents: {len(documents)}")
        save_preprocessed_data(documents, documents_path)
        print("The preprocessed files are saved in:", documents_path)
        return documents

def create_or_load_embeddings(documents, embeddings_path, metadata_path, embedding_model_name):
    """Create or load embeddings."""
    if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
        print(f"Find Embeddings: {embeddings_path}\nFinding Metadata: {metadata_path}")
    else:
        print(f"Creating Embeddings: {embeddings_path}\nCreating Metadata: {metadata_path}")
        embedding_model = SentenceTransformer(embedding_model_name, device=device)
        doc_embeddings = generate_embeddings_long_text(documents, embedding_model, field="content", batch_size=512)
        save_embeddings(doc_embeddings, documents, embeddings_file=embeddings_path, metadata_file=metadata_path)

def create_or_load_bm25(documents, bm25_state_path):
    """Create or load BM25 state."""
    if os.path.exists(bm25_state_path):
        print(f"Find BM25 State: {bm25_state_path}")
    else:
        print(f"Creating BM25 State: {bm25_state_path}")
        tokenized_corpus = preprocess_documents_bm25(documents)
        bm25 = build_bm25_index(tokenized_corpus)
        save_bm25_state(bm25, bm25_state_path, tokenized_corpus)

def create_or_load_chunk_embeddings(documents, embeddings_chunk_path):
    """Create or load chunk embeddings."""
    if os.path.exists(embeddings_chunk_path):
        print(f"Find Embeddings: {embeddings_chunk_path}")
    else:
        print(f"Creating Embeddings with chunk: {embeddings_chunk_path}")
        chunked_embeddings = embedding_documents(documents)
        chunked_embeddings = convert_ndarray_to_list(chunked_embeddings)
        with open(embeddings_chunk_path, "w") as f:
            json.dump(chunked_embeddings, f)

def filter_retrieved_documents(retrieved_docs_with_scores, method, filter_threshold, document_index):
    """Filter retrieved documents based on method and threshold."""
    filtered_inputs = []
    filtered_docs = []
    filtered_docs_ids = []

    for doc in retrieved_docs_with_scores:
        if method == "bm25":
            doc_id_base = doc["doc_id"].split("|")[0]
            if doc_id_base in filtered_docs_ids:
                continue
            filtered_inputs.append({"doc_title": doc["title"], "doc_content": doc["content"]})
            filtered_docs.append({"doc_id": doc["doc_id"], "content": doc["content"]})
            filtered_docs_ids.append(doc_id_base)
        elif method in ["embedding", "chunk"] and doc["cosine_similarity"] > filter_threshold:
            doc_id_base = doc["doc_id"].split("|")[0]
            matched_documents = find_documents_by_id_fast(document_index, doc_id_base)
            doc_content = "".join([d["content"] for d in matched_documents])
            content = {"doc_id": doc_id_base, "content": doc_content}
            if content in filtered_docs:
                continue
            filtered_inputs.append({"doc_title": doc["title"], "doc_content": doc_content})
            filtered_docs.append(content)
            filtered_docs_ids.append(doc_id_base)

    return filtered_inputs, filtered_docs, filtered_docs_ids

def main():
    parser = argparse.ArgumentParser(description="Document Retrieval and Question Answering System")
    parser.add_argument("--data_path", type=str, default="../data", help="Path to the data directory")
    parser.add_argument("--weights_path", type=str, default="../data/weights", help="Path to the weights directory")
    parser.add_argument("--embedding_model_name", type=str, default="all-mpnet-base-v2",
                        help="Name of the embedding model")
    parser.add_argument("--method", type=str, choices=["embedding", "bm25", "chunk"], default="embedding",
                        help="Retrieval method")
    parser.add_argument("--filter_threshold", type=float, default=0.7, help="Threshold for filtering documents")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top documents to retrieve")
    parser.add_argument("--question_title", default="How to configure DC to look on a different App server for the SSP?",
                        type=str,required=True, help="The question to retrieve documents for")
    parser.add_argument("--Llm_model_name", type=str, default="deepseek-chat",
                        help="Name of the Llm model")
    parser.add_argument("--url",default="https://api.deepseek.com/v1",type=str,help="URL to the Llm API")
    parser.add_argument("--token",default="sk-edb1d3602d1d485a88ab517fa1ac1a67",type=str,help="Token to use")

    args = parser.parse_args()

    data_path, weights_path = initialize_paths(args)
    technotes_path = os.path.join(data_path, "training_dev_technotes.sections.json")
    documents_path = os.path.join(data_path, "documents_preprocessed.json")
    embeddings_path = os.path.join(weights_path, "doc_embeddings_all.npy")
    embeddings_chunk_path = os.path.join(weights_path, "processed_embeddings.json")
    metadata_path = os.path.join(weights_path, "doc_metadata_all.json")
    bm25_state_path = os.path.join(weights_path, "bm25_state.json")


    client = get_model({
        "token": args.token,
        "url": args.url,
        "model": args.Llm_model_name,
    })

    embedding_model_name = args.embedding_model_name
    question_title = args.question_title
    method = args.method
    filter_threshold = args.filter_threshold
    top_k = args.top_k

    documents = load_or_preprocess_documents(technotes_path, documents_path)
    create_or_load_embeddings(documents, embeddings_path, metadata_path, embedding_model_name)
    create_or_load_bm25(documents, bm25_state_path)
    create_or_load_chunk_embeddings(documents, embeddings_chunk_path)

    document_index = build_document_index(documents)



    if method == "embedding":
        embedding_model = SentenceTransformer(embedding_model_name, device=device)
        doc_embeddings, documents = load_embeddings(embeddings_file=embeddings_path, metadata_file=metadata_path)
        index = build_faiss_index(doc_embeddings)
        retrieved_docs_with_scores = retrieve_documents_with_scores_new(
            question_title, index, doc_embeddings, documents, embedding_model, top_k=top_k
        )
    elif method == "bm25":
        bm25 = load_bm25_state(bm25_state_path)
        retrieved_docs_with_scores = retrieve_documents_bm25(question_title, bm25, documents, top_k=top_k)
    elif method == "chunk":
        embedding_model = SentenceTransformer(embedding_model_name, device=device)
        processed_results = load_processed_embeddings(embeddings_chunk_path)
        doc_embeddings, documents = fix_emmbedding(processed_results)
        index = build_faiss_index(doc_embeddings)
        retrieved_docs_with_scores = retrieve_documents_with_scores_chunk(
            question_title, index, doc_embeddings, documents, embedding_model, top_k=top_k
        )

    filtered_inputs, filtered_docs, filtered_docs_ids = filter_retrieved_documents(
        retrieved_docs_with_scores, method, filter_threshold, document_index
    )

    input_data = {
        "doc_id": ','.join(filtered_docs_ids),
        "docs": '\n'.join(dict_to_str(d) for d in filtered_inputs)
    }

    if filtered_inputs:
        user_content = prompt_from_question_to_keywords.format(context=input_data['docs'], question=question_title)
        response = llm_generate(user_content=user_content, client=client, cfg=cfg)
        print(f'Answer: {response}')

    else:
        print("I don't have enough information to answer that.")

if __name__ == "__main__":
    main()