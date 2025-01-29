import os
import sys
import json
from IPython.display import display_markdown
from huggingface_hub import login
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image as PIL_Image
import torch
from helper_reflection_img import *
from datetime import datetime

# Main workflow
def main():
    # Step 1: Check GPU and CUDA
    check_gpu_memory()
    check_cuda()

    # Step 2: Load Hugging Face token and model
    load_huggingface_token()
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model, processor = load_model_and_processor(model_name)

    # Step 3: Load an image from the dataset
    current_directory = os.getcwd()
    temp_dir = current_directory
    for _ in range(5):  # Adjust depth as needed
        temp_dir = os.path.dirname(temp_dir)
    data_dir = os.path.join(temp_dir, "data", "piid", "dataset", "original")

    folders = os.listdir(data_dir)
    
    num_prompt = 1
    num_fldr = 4
    num_img = 40

    # Provide prompts
    with open("prompt_kobe.json", "r") as file:
        prompts_dict = json.load(file)

    methods = ['Chain-of-Thought', 'TIDD-EC', 'Few-Shot', 'CO-STAR', 'Zero-Shot', 'Tree-of-Thought']

    for prompt_i in range(num_prompt):
        prmpt_method = methods[prompt_i]
        prompt_text = prompts_dict[prmpt_method]

        for folder_i in range(num_fldr):
            print(f'\n=============== Class: {folder_i + 1} ===============\n')
            fldr_ = folders[folder_i]
            img_fldr_dir = os.path.join(data_dir, fldr_)

            for img_i in range(num_img):
                num_processed_images = 1 + img_i + (folder_i * num_img) + (prompt_i * num_img * num_fldr)
                print(f'\n=============== Processed Images: {num_processed_images}/{num_prompt * num_fldr * num_img} ===============\n')

                img_ = os.listdir(img_fldr_dir)[img_i]
                img_path = os.path.join(img_fldr_dir, img_)
                image = load_image(img_path)

                BASE_GENERATION_SYSTEM_PROMPT = """
                Your task is to generate the best content possible for the user's request. Start your sentences with "ASSISTANT", then directly write your classification.
                If the user provides critique, respond with a revised version of your previous attempt.
                """

                BASE_REFLECTION_SYSTEM_PROMPT = """
                You are a wound specialist tasked with generating critique and recommendations for the user's generated content. Start your sentences with "CRITIQUE".
                Focus on reasoning and do not introduce new information. If the content is flawed or can be improved, output a list of recommendations and critiques. 
                If the content is correct and nothing needs to change, output: <OK>.
                """

                user_prompt = prompt_text
                num_iterations = 2
                max_new_tokens = 4096
                iterative_generation_and_reflection(
                    processor, model, prmpt_method, BASE_GENERATION_SYSTEM_PROMPT, BASE_REFLECTION_SYSTEM_PROMPT, user_prompt, image, num_iterations
                )

if __name__ == "__main__":
    # Get the current time
    current_time = datetime.now()

    # Format the time as sec_min_hr_mm_dd_year
    formatted_time = current_time.strftime("%m_%d_%Y_%H_%M_%S")

    # Generate the file name with timestamp
    file_name = f"workflow_log_{formatted_time}.txt"

    # Redirect stdout to the log file
    sys.stdout = open(file_name, "w")

    # Run the main workflow
    main()

    # Restore default stdout
    sys.stdout.close()
    sys.stdout = sys.__stdout__

    # Notify the user
    print(f"All logs have been saved to {file_name}")
