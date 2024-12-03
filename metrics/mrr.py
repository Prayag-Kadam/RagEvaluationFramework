def mean_reciprocal_rank(relevant_docs_list, retrieved_docs_list):
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
        relevant_docs_list (list): List of sets of relevant document indices for each query.
        retrieved_docs_list (list): List of retrieved document indices for each query.

    Returns:
        float: Mean Reciprocal Rank.
    """
    reciprocal_ranks = []
    for relevant_docs, retrieved_docs in zip(relevant_docs_list, retrieved_docs_list):
        rank = 0
        for idx, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                rank = idx + 1
                break
        reciprocal_ranks.append(1 / rank if rank > 0 else 0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
