import sys
import os
import json

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

# Step 1: Embed queries and documents
model = SentenceTransformer('all-MiniLM-L6-v2')
query_embeddings = model.encode(queries, convert_to_tensor=True)
document_embeddings = model.encode(documents, convert_to_tensor=True)

# Step 2: Compute similarity and rank documents
similarity_scores = util.pytorch_cos_sim(query_embeddings, document_embeddings)
ranked_retrievals = []
for i in range(len(queries)):
    scores = similarity_scores[i].tolist()
    ranked_docs = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
    ranked_retrievals.append(ranked_docs)

# Step 3: Convert answers to relevance labels
relevant_docs_list = [{i} for i in range(len(answers))]

# Step 4: Evaluate metrics
for k in [1, 3, 5]:
    precision = precision_at_k(relevant_docs_list[0], ranked_retrievals[0], k)
    recall = recall_at_k(relevant_docs_list[0], ranked_retrievals[0], k)
    print(f"Precision@{k}: {precision:.2f}")
    print(f"Recall@{k}: {recall:.2f}")

mrr = mean_reciprocal_rank(relevant_docs_list, ranked_retrievals)
print(f"Enhanced Mean Reciprocal Rank: {mrr:.2f}")
