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
    C = defaultdict(lambda: defaultdict(float))
    for term in freq_table.columns:
        for doc in freq_table.index:
            minterm = doc_to_minterm[doc]
            C[term][minterm] += freq_table.loc[doc, term]
    return pd.DataFrame.from_dict(C, orient='index').fillna(0).astype(float)

def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda row: row / np.linalg.norm(row) if np.linalg.norm(row) != 0 else row, axis=1)

def perform_gvsm(documents_dict: dict, query: list) -> pd.DataFrame:
    # Step 1: Frequency table
    N = len(documents_dict)
    documents_dict["query"] = query

    freq_all = build_frequency_table(documents_dict)
    freq_query = freq_all.loc[['query']]
    freq_documents = freq_all.drop(index='query')

    # idf
    idf_documents = (freq_documents > 0).sum(axis=0)
    # print("N",N)
    idf_query = idf_documents.apply(lambda x:0 if x<=0 else np.log10(N / x))
    # print("idf_documents\n",idf_documents)
    # print("idf_query\n",idf_query)

    # Smooth query
    max_freq_query = freq_query.max(axis=1).values[0]
    tf_query = freq_query.copy()
    tf_query[freq_query != 0] = 0.5 + (0.5 * freq_query[freq_query != 0] / max_freq_query)
    tf_idf_query = tf_query * idf_query
    # print("tf_idf_query\n",tf_idf_query)

    # Tfidf the documents
    maxfreq_by_documents = freq_documents.max(axis=1)
    tf_idf_documents = freq_documents.copy()
    for term in tf_idf_documents.columns:
        for index in tf_idf_documents.index:
            tf_idf_documents.loc[index, term] = tf_idf_documents.loc[index, term]/maxfreq_by_documents[index] * idf_query[term]
    # print("doc_tf_idf\n",tf_idf_documents)
    

    # Step 2: Binarize and group documents by minterms
    bin_freq = binarize_frequencies(freq_documents)
    # print("bin_freq\n",bin_freq)
    doc_to_minterm, unique_minterms = assign_minterms(bin_freq)
    # print("doc_to_minterm\n",doc_to_minterm)

    # Step 3: Build term-minterm matrix
    C = build_term_minterm_matrix(tf_idf_documents, doc_to_minterm)
    # print("C",C)
    K = normalize_rows(C)
    # print("K",K)

    # Step 4: Project documents and query into minterm space
    doc_minterm_space = freq_documents.values @ K.values
    query_minterm_space = freq_query.values @ K.values

    # Step 5: Compute cosine similarity
    cos_sim = cosine_similarity(doc_minterm_space, query_minterm_space).flatten()

    # Step 6: Create ranked document table
    ranked_docs = pd.DataFrame({
        "Document": freq_documents.index,
        "CosineSimilarity": cos_sim,
        "Minterm": [doc_to_minterm[doc] for doc in freq_documents.index]
    }).sort_values(by="CosineSimilarity", ascending=False).reset_index(drop=True)

    # Output results
    print("\nRanked Documents by Cosine Similarity:\n", ranked_docs)
    return cos_sim

def main():
    # Input documents and query
    query = ["cat", "cat", "cat", "dog", "dog", "tiger"]
    documents = {
        "D1": ['bird', 'cat', 'bird', 'cat', 'dog', 'dog', 'bird'],
        "D2": ['cat', 'tiger', 'cat', 'dog'],
        "D3": ['dog', 'bird', 'bird'],
        "D4": ['cat', 'tiger'],
        "D5": ['tiger', 'tiger', 'dog', 'tiger', 'cat'],
        "D6": ['cat', 'cat', 'tiger', 'tiger'],
        "D7": ['bird', 'cat', 'dog'],
        "D8": ['dog', 'cat', 'bird'],
        "D9": ['cat', 'dog', 'tiger'],
        "D10": ['tiger', 'cat', 'tiger'],
    }
    res = perform_gvsm(documents, query)
    print('res',res)

if __name__ == "__main__":
    main()