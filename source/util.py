import itertools
from typing import List, Dict
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import json

def load_data(file_path: str):

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def split_text(text, chunk_size=512, overlap=50):
    """
    Split long text into chunks and add overlapping areas。
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
def split_long_text(text, max_tokens=512):
    """
    Split long text into fixed token lengths.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunks.append(" ".join(words[i:i + max_tokens]))
    return chunks

def split_text_semantically(text, max_tokens=512):
    """
    Split text semantically (based on sentences) and aggregate to a length close to max_tokens.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def preprocess_documents(technotes_path: str, include_section=True) -> List[Dict]:

    with open(technotes_path, "r", encoding="utf-8") as file:
        documents = json.load(file)
    processed_docs = []
    for doc in documents.keys():
        doc = documents[doc]
        if include_section:
            sections = doc['sections']
            for section in sections:
                section_id = section["id"]
                section_text = section["text"]
                doc_data = {
                    "id": doc["id"]+"|"+section_id,
                    "title": doc["title"],
                    "content": section_text
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

def save_preprocessed_data(data: List[Dict], output_file: str):

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def convert_ndarray_to_list(data):
    for item in data:
        if isinstance(item["embedding"], np.ndarray):
            item["embedding"] = item["embedding"].tolist()
    return data

def save_embeddings(doc_embeddings, documents, embeddings_file="embeddings.npy", metadata_file="metadata.json"):
    """
    Save embeddings and metadata to files.

    Parameters:
    - doc_embeddings (numpy.ndarray): The embeddings to save.
    - documents (list or dict): The metadata to save.
    - embeddings_file (str): Path to save the embeddings file.
    - metadata_file (str): Path to save the metadata file.
    """
    # Save embeddings to a .npy file
    np.save(embeddings_file, doc_embeddings)

    # Save metadata to a .json file
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)

    print(f"Embeddings and metadata have been saved to {embeddings_file} and {metadata_file}.")

def load_embeddings(embeddings_file="embeddings.npy", metadata_file="metadata.json"):
    """
    Load embeddings and metadata from files.

    Parameters:
    - embeddings_file (str): Path to the embeddings file.
    - metadata_file (str): Path to the metadata file.

    Returns:
    - tuple: A tuple containing the embeddings (numpy.ndarray) and metadata (list or dict).
    """
    # Load embeddings
    doc_embeddings = np.load(embeddings_file)

    # Load metadata
    with open(metadata_file, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # Validate loaded data
    print(f"Data have been loaded from {embeddings_file} and {metadata_file}.")
    print(f"Embeddings type: {type(doc_embeddings)}, shape: {doc_embeddings.shape}")
    print(f"Metadata type: {type(documents)}, number of entries: {len(documents)}")

    return doc_embeddings, documents


def build_document_map(documents):
    return {doc["doc_id"]: doc["content"] for doc in documents}
from collections import defaultdict

def build_document_index(documents):
    index = defaultdict(list)
    for doc in documents:
        id_prefix = doc["doc_id"].split("|")[0]  # 提取 ID 前缀
        index[id_prefix].append(doc)
    return index

def find_documents_by_id_fast(index, id_prefix):
    return index.get(id_prefix, [])

def compute_recall(ref_data, output_data):
    hit = 0.
    hit_true=0.
    na_hit = 0.
    ab_hit = 0.
    ab_hit_true=0.
    total = 0.
    na_total = 0.
    ab_total = 0.
    for (ref, output) in zip(ref_data, output_data):
        ref_id  = ref.split('-')[0]
        output_list = output
        # if '' in output_list:
        #     print(output_list)
        if ref_id == '':
            na_total += 1
        else:
            ab_total += 1
        if ref_id in output_list:
            if ref_id == '':
                na_hit += 1
                hit += 1
            else:
                ab_hit_true+=1
                hit_true += 1
                ab_hit += 1 / len(output_list)
                hit += 1 / len(output_list)
        total += 1
    recall_score = 100 * hit / total
    recall_score_true = 100 * hit_true / total
    na_recall = 100 * na_hit / na_total
    ab_recall = 100 * ab_hit / ab_total
    ab_recall_true = 100 * ab_hit_true / ab_total

    return recall_score,recall_score_true,ab_recall,ab_recall_true

def constraint_embedding_bm25(params):
    """
    Ensure that embeddings and use_bm25 have only two valid combinations:
    1. embeddings=1 and use_bm25=0
    2. embeddings=0 and use_bm25=1
    """
    return (params["embeddings"] == 1 and params["use_bm25"] == 0) or (params["embeddings"] == 0 and params["use_bm25"] == 1)


def generate_ablation_combinations_with_constraints_my(parameters, constraints):
    """
    Generate all combinations of parameters for ablation study, applying constraints.

    Args:
        parameters (dict): A dictionary where keys are parameter names
                           and values are lists of possible values for each parameter.
        constraints (list): A list of constraint functions to filter combinations.

    Returns:
        pd.DataFrame: A DataFrame containing all valid combinations of parameters.
    """
    # Generate all combinations
    keys, values = zip(*parameters.items())
    combinations = list(itertools.product(*values))

    # Apply constraints
    valid_combinations = []
    for combination in combinations:
        param_dict = dict(zip(keys, combination))
        if all(constraint(param_dict) for constraint in constraints):
            valid_combinations.append(combination)

    # Convert to DataFrame for better visualization and usability
    valid_combinations_df = pd.DataFrame(valid_combinations, columns=keys)
    return valid_combinations_df


def generate_markdown_table(parameters, metrics, results):
    """
    Generate a Markdown table for ablation study results.

    Args:
        parameters (list): List of parameter names.
        metrics (list): List of metric names (e.g., ['a', 'b', 'c', 'd']).
        results (list of dict): List of dictionaries containing parameter values and metrics.

    Returns:
        str: A string in Markdown table format.
    """
    # Combine parameters and metrics as column names
    columns = parameters + metrics

    # Convert results into a DataFrame
    df = pd.DataFrame(results, columns=columns)

    # Convert the DataFrame to Markdown
    markdown_table = df.to_markdown(index=False)
    return markdown_table

def dict_to_str(d):
    return f"{d['doc_title']}{d['doc_content']}"