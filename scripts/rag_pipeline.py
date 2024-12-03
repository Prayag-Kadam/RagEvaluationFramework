import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from metrics.mrr import mean_reciprocal_rank
from metrics.precision_recall import precision_at_k, recall_at_k

def rag_pipeline(data_path, model_name, k_values=[1, 3, 5]):
    # Step 1: Load Data
    with open(data_path, "r") as f:
        data = json.load(f)
    queries = data["queries"]
    documents = data["documents"]
    answers = data["answers"]

    # Step 2: Initialize Models
    retriever = SentenceTransformer(model_name)
    generator = pipeline("text-generation", model="gpt2")

    # Step 3: Embed Queries and Documents
    query_embeddings = retriever.encode(queries, convert_to_tensor=True)
    document_embeddings = retriever.encode(documents, convert_to_tensor=True)

    # Step 4: Retrieve Relevant Documents
    similarity_scores = util.pytorch_cos_sim(query_embeddings, document_embeddings)
    retrieval_results = []
    for i, query in enumerate(queries):
        scores = similarity_scores[i].tolist()
        ranked_docs = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
        top_k_docs = [documents[idx] for idx in ranked_docs[:max(k_values)]]
        retrieval_results.append((query, top_k_docs))

    # Step 5: Generate Responses
    generation_results = []
    for query, top_docs in retrieval_results:
        combined_input = f"{query} {top_docs[0]}"  # Use top-1 document for generation
        generated_response = generator(combined_input, max_new_tokens=50)[0]["generated_text"]

        generation_results.append(generated_response)

    # Step 6: Evaluate Metrics
    relevant_docs_list = [{i} for i in range(len(answers))]
    metrics = {"retrieval": [], "generation": []}

    # Retrieval Metrics
    for i, (query, _) in enumerate(retrieval_results):
        ranked_docs = sorted(range(len(similarity_scores[i])), key=lambda x: similarity_scores[i][x], reverse=True)
        for k in k_values:
            precision = precision_at_k(relevant_docs_list[i], ranked_docs, k)
            recall = recall_at_k(relevant_docs_list[i], ranked_docs, k)
            metrics["retrieval"].append({
                "query": query,
                f"Precision@{k}": precision,
                f"Recall@{k}": recall
            })

    # Generation Metrics (Optional)
    from rouge_score import rouge_scorer
    from sacrebleu import corpus_bleu

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    for i, response in enumerate(generation_results):
        rouge_scores = rouge.score(answers[i], response)
        bleu = corpus_bleu([response], [[answers[i]]]).score
        metrics["generation"].append({
            "query": queries[i],
            "response": response,
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
            "bleu": bleu
        })

    return metrics

def save_metrics(metrics, output_path):
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {output_path}")

if __name__ == "__main__":
    data_path = "data/valid_subset.json"
    model_name = "all-MiniLM-L6-v2"
    output_path = "results/rag_metrics.json"

    metrics = rag_pipeline(data_path, model_name)
    save_metrics(metrics, output_path)
