import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import os

# Load the Excel file
file_path = 'result_2024_11_25_12_27_retrieved.xlsx'  # Replace with your Excel file path
df = pd.read_excel(file_path)

print(df.head())  # Check the first few rows
print(df.columns)  # Print column names

# Filter out rows where predicted_class is 0
df = df[df['predicted_class'] != 0]

# Get unique prompt methods
prompt_methods = df['prompt_method'].unique()

# Create directories for saving images
excel_base_name = os.path.splitext(os.path.basename(file_path))[0]
output_dir = 'acc_precision_recall_f1'
subfolder_path = os.path.join(output_dir, excel_base_name)
os.makedirs(subfolder_path, exist_ok=True)

# Store overall metrics
overall_metrics = []

for method in prompt_methods:
    method_df = df[df['prompt_method'] == method]
    
    # Confusion matrix
    cm = confusion_matrix(method_df['target_class'], method_df['predicted_class'])
    
    # Create a new figure for the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                annot_kws={"size": 16})  # Bigger font size for numbers
    
    plt.title(f'Confusion Matrix for {method}', fontsize=16)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('True Class', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Adjust x and y ticks to start from 1
    plt.xticks(ticks=np.arange(len(np.unique(method_df['predicted_class']))), labels=np.arange(1, len(np.unique(method_df['predicted_class'])) + 1))
    plt.yticks(ticks=np.arange(len(np.unique(method_df['target_class']))), labels=np.arange(1, len(np.unique(method_df['target_class'])) + 1))

    # Save the plot as an image in the subfolder
    plt.savefig(os.path.join(subfolder_path, f'confusion_matrix_{method}.png'))
    plt.close()  # Close the plot to free memory
    
    # Calculate overall metrics
    accuracy = accuracy_score(method_df['target_class'], method_df['predicted_class'])
    precision = precision_score(method_df['target_class'], method_df['predicted_class'], average='weighted')
    recall = recall_score(method_df['target_class'], method_df['predicted_class'], average='weighted')
    f1 = f1_score(method_df['target_class'], method_df['predicted_class'], average='weighted')
    
    # Store metrics
    overall_metrics.append((method, accuracy, precision, recall, f1))

    # Plot performance metrics for each method
    plt.figure(figsize=(10, 5))
    metrics = [accuracy, precision, recall, f1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for i, score in enumerate(metrics):
        plt.bar(metric_names[i], score, color='blue' if i == 0 else 'green' if i == 1 else 'orange' if i == 2 else 'red')
    
    # Set title with metrics values
    plt.title(f'{method} Performance Metrics\nAccuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}', fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(subfolder_path, f'performance_metrics_{method}.png'))
    plt.close()

# Find the best model based on accuracy
best_model = max(overall_metrics, key=lambda x: x[1])
best_model_name = best_model[0]
best_accuracy = best_model[1]
best_f1 = best_model[4]

# Plot overall performance metrics
plt.figure(figsize=(10, 5))
x_labels = []
markers = ['o', 's', '^', 'x']

for i, (method, accuracy, precision, recall, f1) in enumerate(overall_metrics):
    x_labels.append(method)
    plt.scatter(method, accuracy, marker=markers[0], label='Accuracy' if i == 0 else "", color='blue')
    plt.scatter(method, precision, marker=markers[1], label='Precision' if i == 0 else "", color='green')
    plt.scatter(method, recall, marker=markers[2], label='Recall' if i == 0 else "", color='orange')
    plt.scatter(method, f1, marker=markers[3], label='F1 Score' if i == 0 else "", color='red')

# Set title with best model info
plt.title(f'Overall Performance Metrics by Prompt Method\nBest Model: {best_model_name} (Accuracy: {best_accuracy:.2f}, F1: {best_f1:.2f})', fontsize=16)
plt.xlabel('Prompt Method', fontsize=12)  # Reduced font size for x-axis label
plt.ylabel('Score', fontsize=14)
plt.xticks(rotation=45, fontsize=10)  # Smaller font size for x-axis tick labels
plt.ylim(0, 1)
plt.legend()
plt.savefig(os.path.join(subfolder_path, 'overall_performance_metrics.png'))
plt.close()

# Print overall metrics
metrics_df = pd.DataFrame(overall_metrics, columns=['Prompt Method', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
print(metrics_df)



u=1