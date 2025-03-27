import random
import os
import json

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

datasettamount = 1000

min_doc_amount_in_dataset = 2
max_doc_amount_in_dataset = 20
min_word_in_doc = 1
max_word_in_doc = 7
word_set = ["bird", "cat", "dog", "tiger"]
min_query_size = 1
max_query_size = 6

def generate_unique_shuffles(count, min_size, max_size, words, exclude=set()):
    seen = set(exclude)
    results = []

    while len(results) < count:
        size = random.randint(min_size, max_size)
        combo = tuple(random.sample(words * ((size // len(words)) + 1), size))
        if combo not in seen:
            seen.add(combo)
            results.append(combo)

    return results

if __name__ == "__main__":
    for dataset_index in range(datasettamount):
        # Generate documents
        documents = generate_unique_shuffles(random.randint(min_doc_amount_in_dataset,max_doc_amount_in_dataset), min_word_in_doc, max_word_in_doc, word_set)

        # Generate query, unique from documents
        query = generate_unique_shuffles(1, min_query_size, max_query_size, word_set, exclude=set(documents))[0]

        from gvsm import perform_gvsm
        gvsm_ranked_docs = perform_gvsm({str(i): doc for i, doc in enumerate(documents)}, query)

        # Prepare JSON data
        json_data = {
            "documents": [list(doc) for doc in documents],
            "query": list(query),
            "gvsm_ranked_docs": gvsm_ranked_docs.tolist()
        }

        # Save to file
        json_path = os.path.join(output_dir, f"dataset_{dataset_index}.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
