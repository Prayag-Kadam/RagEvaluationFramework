import pandas as pd
import os
import json

def preprocess_dataset(dataset_path, sample_size=5):
    """
    Preprocess the dataset to extract a manageable subset for evaluation.

    Args:
        dataset_path (str): Path to the CSV file containing the dataset.
        sample_size (int): Number of samples to extract.

    Returns:
        dict: Subset containing queries, passages, and ground truth answers.
    """
    # Load the dataset
    data = pd.read_csv(dataset_path)
    
    # Sample a subset of data
    subset = data.sample(n=sample_size, random_state=42)
    
    # Format the subset into a dictionary
    processed_subset = {
        "queries": subset["query"].tolist(),
        "documents": subset["finalpassage"].tolist(),
        "answers": subset["answers"].tolist(),
    }
    return processed_subset

def save_subset(subset, output_path):
    """
    Save the processed subset to a JSON file.

    Args:
        subset (dict): Processed subset of data.
        output_path (str): Path to save the JSON file.
    """
    # Create the directory if it doesn't exist
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the subset as JSON
    with open(output_path, "w") as f:
        json.dump(subset, f, indent=4)

# Paths to dataset files
train_path = "D:/EvaluationFramework/data/train.csv"
valid_path = "D:/EvaluationFramework/data/valid.csv"

# Preprocess the datasets
train_subset = preprocess_dataset(train_path, sample_size=5)
valid_subset = preprocess_dataset(valid_path, sample_size=5)

# Save the subsets
save_subset(train_subset, "D:/EvaluationFramework/data/train_subset.json")
save_subset(valid_subset, "D:/EvaluationFramework/data/valid_subset.json")

# Display the results
print("Train Subset:", train_subset)
print("Validation Subset:", valid_subset)
