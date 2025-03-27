import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def build_frequency_table(documents: dict) -> pd.DataFrame:
    doc_names = list(documents.keys())
    corpus = [' '.join(words) for words in documents.values()]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return pd.DataFrame(X.toarray(), index=doc_names, columns=vectorizer.get_feature_names_out())

def binarize_frequencies(freq_table: pd.DataFrame) -> pd.DataFrame:
    return freq_table.applymap(lambda x: int(x > 0))

def assign_minterms(freq_table_bin: pd.DataFrame) -> dict:
    unique_patterns = freq_table_bin.drop_duplicates()
    doc_to_minterm = {}
    for doc, row in freq_table_bin.iterrows():
        for i, (index, pattern) in enumerate(unique_patterns.iterrows(), 1):
            if (row == pattern).all():
                doc_to_minterm[doc] = i
                break
    return doc_to_minterm, unique_patterns

def build_term_minterm_matrix(freq_table: pd.DataFrame, doc_to_minterm: dict) -> pd.DataFrame:
    C = defaultdict(lambda: defaultdict(int))
    for term in freq_table.columns:
        for doc in freq_table.index:
            minterm = doc_to_minterm[doc]
            C[term][minterm] += freq_table.loc[doc, term]
    return pd.DataFrame.from_dict(C, orient='index').fillna(0).astype(int)

def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda row: row / np.linalg.norm(row) if np.linalg.norm(row) != 0 else row, axis=1)

def perform_gvsm(documents: dict, query: list) -> pd.DataFrame:
    # Step 1: Frequency table
    documents["query"] = query
    freq_table = build_frequency_table(documents)
    query_freq = freq_table.loc[['query']]
    doc_freq = freq_table.drop(index='query')

    # Step 2: Binarize and group documents by minterms
    bin_freq = binarize_frequencies(doc_freq)
    doc_to_minterm, unique_minterms = assign_minterms(bin_freq)

    # Step 3: Build term-minterm matrix
    term_minterm_matrix = build_term_minterm_matrix(doc_freq, doc_to_minterm)
    K = normalize_rows(term_minterm_matrix)

    # Step 4: Project documents and query into minterm space
    doc_minterm_space = doc_freq.values @ K.values
    query_minterm_space = query_freq.values @ K.values

    # Step 5: Compute cosine similarity
    cos_sim = cosine_similarity(doc_minterm_space, query_minterm_space).flatten()

    # Step 6: Create ranked document table
    ranked_docs = pd.DataFrame({
        "Document": doc_freq.index,
        "CosineSimilarity": cos_sim,
        "Minterm": [doc_to_minterm[doc] for doc in doc_freq.index]
    }).sort_values(by="CosineSimilarity", ascending=False).reset_index(drop=True)

    # Output results
    print("\nRanked Documents by Cosine Similarity:\n", ranked_docs)
    return cos_sim

def main():
    # Input documents and query
    query = ["cat", "cat", "cat", "dog", "dog", "tiger"]
    documents = {
        "D1": ["bird", "cat", "bird", "cat", "dog", "dog", "bird"],
        "D2": ["cat", "tiger", "cat", "dog"],
        "D3": ["dog", "bird", "bird"],
        "D4": ["cat", "tiger"],
        "D5": ["tiger", "tiger", "dog", "tiger", "cat"],
        "D6": ["cat", "cat", "tiger", "tiger"],
        "D7": ["bird", "bird", "cat", "dog"],
        "D8": ["cat", "cat", "cat", "dog", "tiger"],
        "D9": ["cat", "cat", "tiger", "tiger", "tiger", "tiger"],
        "D10": ["dog", "dog", "bird"],
        "query": query
    }
    res = perform_gvsm(documents, query)
    print('res',res)

if __name__ == "__main__":
    main()