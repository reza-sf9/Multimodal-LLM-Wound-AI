# Implementation is based on : https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct


# limitation of Llama: 
# Llama only support one image per prompt, but can have multi-run 


# NOTE: this code now working well with wound images 
# I can do compare of results of Llama 3.2 with GPT-4o

from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image as PIL_Image
import json 
from huggingface_hub import login
import os
import datetime
from helper_llama import *
from tqdm import tqdm


img_per_cls = 200


# Get current time (year_month_day_hour_minute)
now = datetime.datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M")

saved_file = 'result_%s.xlsx' % current_time

print("Logging in to Hugging Face")
config_file = "hf_rsf_llama_key.json"

# Load the token from the JSON file
with open(config_file, "r") as f:
    config = json.load(f)

hf_token = config.get("huggingface_token")
login(hf_token)


# load prompt dict 
json_file_path = "path_to_your_json_file.json"  # Replace with the path to your JSON file

with open("prompt_kobe.json", "r") as file:
    prompts_dict = json.load(file)

# load model
print("Loading Llama model")
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
# model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"

# check if gpu is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)

# Ensure the device is set to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model = model.to(device)

# Check the device of the model
print(f"\n\nModel is on: {next(model.parameters()).device}")

print(f"check if model allocated on cuda memory:  {torch.cuda.memory_allocated()}")
print(f"check if memory is reserved on cuda:      {torch.cuda.memory_reserved()}")

# find out where the model is saved
from transformers.utils.hub import TRANSFORMERS_CACHE

print(f"\n\nTransformers cache directory: {TRANSFORMERS_CACHE}")


# methods = ['CO-STAR', 'Zero-Shot', 'Fewv-Shot', 'TIDD-EC', 'Chain-of-Thought', 'Tree-of-Thought']
methods = ['CO-STAR']

all_results = []  # List to accumulate all results


# load image 
# get current directory
current_directory = os.getcwd()

temp_dir = current_directory
for i in range(3):
    temp_dir = os.path.dirname(temp_dir)

data_dir = os.path.join(temp_dir, "data", "piid", "dataset", "original")

# get all folders name in the data_dir 
folders = os.listdir(data_dir)
# folders = ['1']

# Initialize total images to process
img_num_to_process = len(folders) * img_per_cls * len(methods)
img_num_processed = 0

print(f"\n\nTotal images to process: {img_num_to_process}\n\n")

# Initialize tqdm progress bar
with tqdm(total=img_num_to_process, desc="Processing Images", unit="image") as pbar:
    for fldr in folders:
        # Convert folder name to class label
        cls_ = int(fldr)
        img_fldr_dir = os.path.join(data_dir, fldr)

        # Get all images in the folder
        images = os.listdir(img_fldr_dir)

        for i in range(img_per_cls):
            img = images[i]
            img_dir = os.path.join(img_fldr_dir, img)

            for prmpt_method in methods:
                img_num_processed += 1

                # Update the progress bar
                pbar.update(1)

                # PROMPT
                prompt_text = prompts_dict[prmpt_method]

                # Prepare the prompt
                mssg = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image"},
                        ],
                    }
                ]


                # debug
                print('\n\n\n')
                print(f"msg: {mssg}")
                print('\n\n\n')


                input_prompt = processor.apply_chat_template(
                    mssg, add_generation_prompt=True
                )

                # debug 
                print('\n\n\n')
                print(f"input_prompt: {input_prompt}")
                print('\n\n\n')

                image = PIL_Image.open(img_dir).convert("RGB")
                inputs = processor(
                    text=input_prompt,
                    images=image,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(model.device)

                prompt_len = len(inputs["input_ids"][0])
                output = model.generate(
                    **inputs,
                    max_new_tokens=256,
                )

                generated_tokens = output[:, prompt_len:]
                result = [img, cls_, prmpt_method, processor.decode(generated_tokens[0])]

                all_results.append(result)

# Save all results to the Excel file after processing all images
save_results_to_excel(saved_file, all_results, model_id)

print(f"Results saved to {saved_file}")


u=1