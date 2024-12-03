import json

# Load train_subset from the JSON file
with open("D:/EvaluationFramework/data/train_subset.json", "r") as f:
    train_subset = json.load(f)


def precision_at_k(relevant_docs, retrieved_docs, k):
    """
    Calculate Precision@K.

    Args:
        relevant_docs (set): Set of relevant document indices.
        retrieved_docs (list): List of retrieved document indices.
        k (int): Number of top documents to consider.

    Returns:
        float: Precision@K.
    """
    top_k_retrieved = retrieved_docs[:k]
    relevant_retrieved = set(top_k_retrieved).intersection(relevant_docs)
    return len(relevant_retrieved) / k


def recall_at_k(relevant_docs, retrieved_docs, k):
    """
    Calculate Recall@K.

    Args:
        relevant_docs (set): Set of relevant document indices.
        retrieved_docs (list): List of retrieved document indices.
        k (int): Number of top documents to consider.

    Returns:
        float: Recall@K.
    """
    top_k_retrieved = retrieved_docs[:k]
    relevant_retrieved = set(top_k_retrieved).intersection(relevant_docs)
    return len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0


# Example data
relevant_docs = {0, 1}  # Example indices of relevant documents
retrieved_docs = [0, 2, 3, 1]  # Simulated retrieval results (top-4 documents)

# Calculate Precision@K and Recall@K
k = 3
precision = precision_at_k(relevant_docs, retrieved_docs, k)
recall = recall_at_k(relevant_docs, retrieved_docs, k)

print(f"Precision@{k}: {precision:.2f}")
print(f"Recall@{k}: {recall:.2f}")

# Assume train_subset has been preprocessed
queries = train_subset["queries"]
documents = train_subset["documents"]
answers = train_subset["answers"]

# Simulate retrieval (for simplicity, using document indices)
retrieved_docs = list(range(len(documents)))  # Top documents by order

# Evaluate for each query
for i, query in enumerate(queries):
    relevant_docs = {i}  # Assume only one correct answer per query
    print(f"Query: {query}")
    for k in [1, 3, 5]:  # Evaluate for multiple K values
        precision = precision_at_k(relevant_docs, retrieved_docs, k)
        recall = recall_at_k(relevant_docs, retrieved_docs, k)
        print(f"  Precision@{k}: {precision:.2f}")
        print(f"  Recall@{k}: {recall:.2f}")
