import pandas as pd
import re
import os

# Load the original Excel file
excel_file_name = "result_2024_11_25_12_27.xlsx"  # Specify your Excel file
file_path = os.path.join('results_from_llama', excel_file_name)

df = pd.read_excel(file_path)

num_char = 50
# Initialize a list to store the predicted classes
predicted_classes = []

# Process classification results in column D starting from row 4 (index 3)
for index, row in df.iterrows():
    print(f"Processing row {index}...")
    if index >= 3:  # Start processing from row 4
        result = str(row[3])  # Column D is index 3
        first_30_chars = result[:num_char]
        last_30_chars = result[-num_char:]

        # Check for "Stage X" in both first and last 50 characters
        match_first = re.search(r'Stage (\d)', first_30_chars)
        match_last = re.search(r'Stage (\d)', last_30_chars)

        if match_first:
            predicted_classes.append(int(match_first.group(1)))  # Assign the number as class
        elif match_last:
            predicted_classes.append(int(match_last.group(1)))  # Assign the number as class
        else:
            predicted_classes.append(0)  # Assign class 0 if not found
        

# Create a new DataFrame for the output
output_df = pd.DataFrame({
    'img_name': df.iloc[3:, 0].values,                # Add the original file name
    'prompt_method': df.iloc[3:, 2].values,     # Column C (index 2) from row 4 down
    'target_class': df.iloc[3:, 1].values,       # Column B (index 1) from row 4 down
    'predicted_class': pd.Series(predicted_classes)
})

a = df.iloc[3:, 0].values

# Create the new Excel file name
base_name, ext = os.path.splitext(file_path)
new_file_path = f"{base_name}_retrieved{ext}"

# Save the new DataFrame to the new Excel file
output_df.to_excel(new_file_path, index=False)

print(f"Predicted classes have been written to the new Excel file: {new_file_path}")

u=1
