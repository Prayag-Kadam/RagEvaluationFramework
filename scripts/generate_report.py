import json
import sys
import os

# Load comparison results
with open("results/model_comparison.json", "r") as f:
    results = json.load(f)

# Generate a summary report
report_lines = []
report_lines.append("Model Comparison Report\n")
report_lines.append("=" * 30 + "\n")

for result in results:
    report_lines.append(f"Model: {result['model']}\n")
    report_lines.append(f"Mean Reciprocal Rank (MRR): {result['MRR']:.2f}\n")
    for k in [1, 3, 5]:
        report_lines.append(f"Precision@{k}: {result[f'Precision@{k}']:.2f}\n")
        report_lines.append(f"Recall@{k}: {result[f'Recall@{k}']:.2f}\n")
    report_lines.append("-" * 30 + "\n")

# Save the report to a file
report_path = "results/comparison_report.txt"
with open(report_path, "w") as f:
    f.writelines(report_lines)

print(f"Report generated and saved to '{report_path}'.")
