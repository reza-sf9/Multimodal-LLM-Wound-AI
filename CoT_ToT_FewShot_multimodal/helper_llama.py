import os
import pandas as pd
from torchvision import transforms
from PIL import Image as PIL_Image

# Save results to an Excel file
import pandas as pd
import os

def save_results_to_excel(excel_file, data, model_info):
    # Create results directory if it doesn't exist
    results_dir = 'results_from_llama'
    os.makedirs(results_dir, exist_ok=True)

    # Set the full path for the Excel file
    full_excel_path = os.path.join(results_dir, excel_file)

    # Check if there is a file with the name of full_excel_path, remove it if it exists
    if os.path.isfile(full_excel_path):
        os.remove(full_excel_path)

    # Convert data into a pandas DataFrame
    df = pd.DataFrame(data, columns=['image_name', 'target_class', 'prompt_method', 'model_result'])

    # Ensure model_info has all necessary keys or values
    if isinstance(model_info, dict):
        model_info_values = [
            model_info.get('model', 'Unknown'),
            model_info.get('max_tokens', 'Unknown'),
            model_info.get('num_img_per_class', 'Unknown'),
        ]
    else:
        # Assume it's a single string (model ID); set placeholders for others
        model_info_values = [model_info, 'Unknown', 'Unknown']

    # Create a DataFrame for the model info
    model_df = pd.DataFrame([model_info_values], columns=['model', 'max_tokens', 'num_img_per_class'])

    # Create a new Excel file and write the model info
    with pd.ExcelWriter(full_excel_path, engine='openpyxl') as writer:
        # Write model info as a single row
        model_df.to_excel(writer, index=False, header=True, startrow=0)

        # Write the image results starting from the next row
        df.to_excel(writer, index=False, header=True, startrow=len(model_df) + 2)

    print(f"Results saved to {full_excel_path}")


# CUDA-enabled image preprocessing
def preprocess_image_cuda(image_path, device):
    # Load and return the PIL image directly
    image = PIL_Image.open(image_path).convert("RGB")

    # Define image transformations (convert to tensor without normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to PyTorch tensor (C, H, W) normalized to [0, 1]
    ])
    
    # Apply transformations and move the tensor to GPU
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return image_tensor
