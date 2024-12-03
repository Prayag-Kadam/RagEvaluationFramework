import json
import sys
import os

# Add the project root directory to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentence_transformers import SentenceTransformer, util
from metrics.mrr import mean_reciprocal_rank
from metrics.precision_recall import precision_at_k, recall_at_k

# Load dataset
with open("data/train_subset.json", "r") as f:
    train_subset = json.load(f)

queries = train_subset["queries"]
documents = train_subset["documents"]
answers = train_subset["answers"]

# Convert answers to relevant document indices
relevant_docs_list = [{i} for i in range(len(answers))]

# Define models to compare
models = {
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "multi-qa-mpnet-base-dot-v1": "multi-qa-mpnet-base-dot-v1"
}

# Initialize results
results = []

for model_name, model_path in models.items():
    # Load model
    model = SentenceTransformer(model_path)

    # Generate embeddings
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    document_embeddings = model.encode(documents, convert_to_tensor=True)

    # Compute similarity scores and rank documents
    similarity_scores = util.pytorch_cos_sim(query_embeddings, document_embeddings)
    ranked_retrievals = []
    for i in range(len(queries)):
        scores = similarity_scores[i].tolist()
        ranked_docs = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
        ranked_retrievals.append(ranked_docs)

    # Calculate metrics
    mrr = mean_reciprocal_rank(relevant_docs_list, ranked_retrievals)
    metrics = {
        "model": model_name,
        "MRR": mrr
    }
    for k in [1, 3, 5]:
        metrics[f"Precision@{k}"] = precision_at_k(relevant_docs_list[0], ranked_retrievals[0], k)
        metrics[f"Recall@{k}"] = recall_at_k(relevant_docs_list[0], ranked_retrievals[0], k)

    results.append(metrics)

# Save results to file
with open("results/model_comparison.json", "w") as f:
    json.dump(results, f, indent=4)

print("Model comparison completed. Results saved to 'results/model_comparison.json'.")
