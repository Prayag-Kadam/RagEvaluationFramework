# **Final Report: RAG Evaluation Framework**

## **Overview**
This project implements a comprehensive framework for evaluating retrieval-augmented generation (RAG) systems. It focuses on assessing both retrieval quality and generation quality, integrating them into an end-to-end pipeline. The implementation is aimed at benchmarking and providing insights into the performance of retrieval and generation components for GenAI applications.

---

## **Approach**

### **Task 1: Retrieval System Evaluation**
1. **Objective**:
   - Evaluate retrieval models using key metrics: Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).

2. **Implementation**:
   - Two embedding models, `all-MiniLM-L6-v2` and `multi-qa-mpnet-base-dot-v1`, were used for retrieval.
   - Queries and documents were embedded using SentenceTransformers, and cosine similarity scores were computed.
   - Top-K documents were ranked, and metrics were calculated for K=1, 3, and 5.

3. **Results**:
   - Both models performed identically on the dataset due to its simplicity:
     - **MRR**: 1.00
     - **Precision@1**: 1.00
     - **Recall@5**: 1.00
   - Observations highlighted the dataset's limitations in differentiating model performance.

4. **Challenges**:
   - Single relevant document per query limited diversity.
   - A larger dataset with multiple relevance levels would better evaluate model differences.

---

### **Task 2: RAG System Evaluation**
1. **Objective**:
   - Create a unified evaluation module to assess both retrieval and generation stages.

2. **Implementation**:
   - A pipeline (`rag_pipeline.py`) was developed to integrate retrieval and generation.
   - **Retrieval Metrics**:
     - Precision@K, Recall@K, and MRR were computed as in Task 1.
   - **Generation Metrics**:
     - ROUGE, BLEU, and BERTScore were calculated to evaluate generated responses.
     - GPT-2 was used as the generation model.
   - Combined metrics provided insights into the system's overall coherence and relevance.

3. **Results**:
   - Retrieval metrics showed consistent performance across queries.
   - Generation metrics highlighted variations in response quality, with BLEU and ROUGE scores reflecting alignment with ground truth.

4. **Challenges**:
   - Handling long queries and truncation during generation required careful tuning.
   - The generation model occasionally produced irrelevant responses due to dataset context limitations.

---

## **Key Learnings**
1. **Effectiveness of Metrics**:
   - Precision@K and Recall@K effectively captured retrieval relevance trends.
   - ROUGE and BLEU were useful for quantifying generation quality, but subjective human evaluation could add value.

2. **Model Performance**:
   - Both retrieval models demonstrated excellent semantic understanding for the given dataset.
   - Fine-tuning GPT-2 could improve response relevance and coherence.

3. **Integration Insights**:
   - Combining retrieval and generation allowed for a holistic evaluation of RAG systems.
   - A well-designed pipeline ensures scalability for more complex datasets or tasks.

---

## **Future Improvements**
1. **Dataset**:
   - Incorporate larger, more diverse datasets (e.g., full MS MARCO, Natural Questions).
   - Use queries with multiple relevance levels to highlight model differences.

2. **Advanced Tools**:
   - Explore frameworks like **TruLens** or **Confident-AI** for advanced evaluations.
   - Integrate multimodal capabilities (e.g., text, images, videos) for richer RAG evaluation.

3. **Model Enhancements**:
   - Fine-tune GPT-2 or use domain-specific generation models for improved responses.
   - Experiment with state-of-the-art retrieval models like `GTR-XL`.

4. **User Feedback**:
   - Add mechanisms to collect user feedback for iterative improvements.

---

## **Challenges Faced**
1. Dataset simplicity made it difficult to showcase model differences.
2. Managing long input sequences required truncation strategies.
3. Limited time constrained exploration of advanced tools and techniques.

---

## **Submission Details**
- **Repository Link**: [Insert your private GitHub repository link here]
- **Collaborators Added**: 
  - `Anandtoshniwal`
  - `ajv009`
- **Deliverables**:
  - Comprehensive evaluation pipeline.
  - Metrics saved in structured files (`rag_metrics.json`).
  - Summary reports (`comparison_report.txt`).
  - Clear documentation (`README.md`).

---

