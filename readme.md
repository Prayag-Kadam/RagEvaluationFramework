# **RAG Evaluation Framework**

## **Overview**
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to evaluate retrieval and generation systems. The framework uses retrieval metrics (e.g., Precision@K, Recall@K, MRR) and generation metrics (e.g., ROUGE, BLEU, BERTScore) to provide a comprehensive analysis of model performance. It is designed to support both retrieval system evaluation and end-to-end RAG pipeline analysis.

---

## **Pipeline Workflow**

### **Task 1: Retrieval System Evaluation**
#### **Objective**
Evaluate the performance of retrieval systems using key metrics and compare two state-of-the-art embedding models.

#### **Implementation Details**
1. **Metrics**:
   - **Precision@K**: Proportion of relevant documents in the top K retrieved.
   - **Recall@K**: Proportion of all relevant documents retrieved within the top K.
   - **Mean Reciprocal Rank (MRR)**: Indicates how early the first relevant document appears in the ranked list.

2. **Models**:
   - **all-MiniLM-L6-v2**: Lightweight embedding model optimized for semantic tasks.
   - **multi-qa-mpnet-base-dot-v1**: High-performing model for multi-question answering and semantic search.

3. **Dataset**:
   - Subset of the **MS MARCO dataset**.
   - 5 queries, each with one relevant document and associated passages.

#### **Results**
| Metric              | all-MiniLM-L6-v2 | multi-qa-mpnet-base-dot-v1 |
|---------------------|------------------|----------------------------|
| MRR                 | 1.00             | 1.00                       |
| Precision@1         | 1.00             | 1.00                       |
| Recall@1            | 1.00             | 1.00                       |
| Precision@3         | 0.33             | 0.33                       |
| Recall@3            | 1.00             | 1.00                       |
| Precision@5         | 0.20             | 0.20                       |
| Recall@5            | 1.00             | 1.00                       |

#### **Key Observations**
- **Model Performance**:
  - Both models achieved perfect MRR, with the first relevant document always ranked first.
  - High recall across all queries indicates effective retrieval.
- **Dataset Simplicity**:
  - The single relevant document per query limited differentiation between models.
  - Using larger datasets or queries with multiple relevance levels could improve evaluation.
- **Metric Effectiveness**:
  - Precision@K and Recall@K revealed trends across K values.
  - MRR highlighted the ranking quality of relevant documents.

#### **Challenges**
- Dataset simplicity limited the ability to showcase model differences.
- Evaluation was restricted to two models due to time constraints.

---

### **Task 2: RAG Pipeline**
#### **Objective**
Evaluate the end-to-end RAG system by combining retrieval and generation processes.

#### **Workflow**
1. **Preprocessing**:
   - Extract subsets of queries, documents, and answers for evaluation.
   - Script: `scripts/preprocess.py`.

2. **Retrieval**:
   - Rank documents based on semantic similarity using embedding models.
   - Metrics: Precision@K, Recall@K, and MRR.
   - Script: `scripts/evaluate.py`.

3. **Generation**:
   - Generate responses using the top-retrieved documents.
   - Metrics: ROUGE, BLEU, and BERTScore.
   - Model: `GPT-2` (default).
   - Script: `scripts/rag_pipeline.py`.

4. **Evaluation**:
   - Save detailed metrics to `results/rag_metrics.json`.
   - Summarize results in `results/comparison_report.txt`.

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/RagEvaluationFramework.git
   cd RagEvaluationFramework
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the directory structure is intact:
   ```
   RagEvaluationFramework/
   ├── data/
   ├── metrics/
   │   ├── mrr.py
   │   ├── precision_recall.py
   ├── results/
   │   ├── rag_metrics.json
   │   ├── comparison_report.txt
   ├── scripts/
   │   ├── preprocess.py
   │   ├── evaluate.py
   │   ├── rag_pipeline.py
   │   ├── generate_report.py
   ├── README.md
   ├── requirements.txt
   ```

---

## **Usage**
### **1. Preprocess Data**
Prepare subsets for training and validation:
```bash
python scripts/preprocess.py
```
- Outputs: `data/train_subset.json`, `data/valid_subset.json`.

### **2. Evaluate Retrieval**
Compute retrieval metrics for the models:
```bash
python scripts/evaluate.py
```
- Outputs metrics directly to the console.

### **3. Run the RAG Pipeline**
Combine retrieval and generation into a unified evaluation:
```bash
python scripts/rag_pipeline.py
```
- Saves detailed metrics to `results/rag_metrics.json`.

### **4. Generate a Summary Report**
Create a human-readable summary:
```bash
python scripts/generate_report.py
```
- Report saved to `results/comparison_report.txt`.

---

## **Metrics Explanation**
### **Retrieval Metrics**
- **Precision@K**: Measures the fraction of relevant documents among the top K retrieved.
- **Recall@K**: Measures the fraction of all relevant documents retrieved within the top K.
- **MRR**: Provides an average reciprocal rank of the first relevant document across queries.

### **Generation Metrics**
- **ROUGE**: Measures overlap of n-grams between generated responses and ground-truth answers.
- **BLEU**: Measures fluency by computing n-gram overlap precision.
- **BERTScore**: Uses embeddings to measure semantic similarity between generated and ground-truth answers.

---

## **Future Work**
1. **Dataset Enhancement**:
   - Use larger, more diverse datasets with queries having multiple relevance levels.
2. **Model Expansion**:
   - Include other embedding models like `GTR-XL` or `DistilBERT`.
3. **RAG Refinement**:
   - Fine-tune generation models for domain-specific tasks.
   - Incorporate multimodal data (e.g., text, images, videos).

---

## **Acknowledgements**
- **Hugging Face Transformers**: For generation models and pipelines.
- **Sentence Transformers**: For efficient retrieval with embeddings.
- **SacreBLEU**: For BLEU score evaluation.
