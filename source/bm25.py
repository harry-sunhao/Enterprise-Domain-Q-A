import json

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_documents_bm25(documents, field="content"):
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()

    tokenized_corpus = []
    for doc in documents:
        tokens = word_tokenize(doc[field].lower())
        filtered_tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
        tokenized_corpus.append(filtered_tokens)
    return tokenized_corpus


def build_bm25_index(tokenized_corpus,k1=1.5,b=0.75):
    bm25 = BM25Okapi(tokenized_corpus,k1=k1,b=b)
    return bm25


def retrieve_documents_bm25(query, bm25, documents, top_k=3, score_threshold=0.1):

    query_tokens = word_tokenize(query.lower())
    scores = bm25.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    retrieved_docs_with_scores = []
    for idx in ranked_indices:
        score = scores[idx]
        if score < score_threshold:
            continue

        retrieved_docs_with_scores.append({
            "doc_id": documents[idx]["doc_id"],
            "title": documents[idx]["title"],
            "content": documents[idx]["content"],
            "score": score
        })

    return retrieved_docs_with_scores

def save_bm25_state(bm25, file_path,tokenized_corpus ):
    state = {
        "idf": bm25.idf,
        "doc_len": bm25.doc_len,
        "average_doc_len": bm25.avgdl,
        "corpus": tokenized_corpus
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(state, f)

def load_bm25_state(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    bm25 = BM25Okapi(state["corpus"])
    bm25.idf = state["idf"]
    bm25.doc_len = state["doc_len"]
    bm25.avgdl = state["average_doc_len"]
    return bm25