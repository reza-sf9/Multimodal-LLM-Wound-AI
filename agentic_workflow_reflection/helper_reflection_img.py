
import json
from IPython.display import display_markdown
from huggingface_hub import login
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image as PIL_Image
import torch

# Function to check GPU memory usage
def check_gpu_memory():
    """
    Prints the total, reserved, allocated, and free memory of the GPU.
    """
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    reserved_memory = torch.cuda.memory_reserved(0) / 1024**3
    allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
    free_memory = total_memory - allocated_memory  # Total free memory on GPU

    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"Reserved Memory: {reserved_memory:.2f} GB")
    print(f"Allocated Memory: {allocated_memory:.2f} GB")
    print(f"Free Memory: {free_memory:.2f} GB\n")

# Function to ensure CUDA is available
def check_cuda():
    """
    Verifies CUDA availability and prints GPU details.
    Raises an error if CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available. Ensure PyTorch is installed with CUDA support.")
    print("CUDA is available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))

# Function to load Hugging Face token
def load_huggingface_token(config_file="hf_rsf_llama_key.json"):
    """
    Loads Hugging Face API token from a JSON configuration file.
    """
    with open(config_file, "r") as f:
        config = json.load(f)
    login(config.get("huggingface_token"))

# Function to load the model and processor
def load_model_and_processor(model_name):
    """
    Loads the LLaMA Vision model and processor. Attempts to load the model fully on GPU.
    Falls back to auto device mapping if GPU memory is insufficient.
    """
    processor = AutoProcessor.from_pretrained(model_name)
    try:
        print("Attempting to load model entirely on GPU...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=None
        )
        model.to("cuda")  # Explicitly move the model to GPU
    except RuntimeError as e:
        print("Failed to load model entirely on GPU. Error:", e)
        print("Falling back to auto device map.")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    return model, processor

# Function to load an image
def load_image(image_path):
    """
    Opens and converts an image to RGB format.
    """
    return PIL_Image.open(image_path).convert("RGB")

# Function to prepare input for model generation
def prepare_input(processor, prompt, image):
    """
    Prepares input tensors for the model using the processor and given prompt.
    """

    # dubug
    # print('\n\n\n')
    # print(f"Prompt: {prompt}")  
    # print('\n\n\n')

    
    input_prompt = processor.apply_chat_template(
        prompt, add_generation_prompt=True)
    
    # dubug
    # print('\n\n\n')
    # print(f"Input Prompt: {input_prompt}")
    # print('\n\n\n')
    
    inputs = processor(
                    text=input_prompt,
                    images=image,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to("cuda")

    return inputs

# Function to generate output from the model
def generate_output(model, processor, inputs, max_new_tokens):
    """
    Generates output from the model and decodes it using the processor.
    """
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(outputs[0], skip_special_tokens=True)

# Function to remove repeated lines from generated content
def clean_generated_content(gen_content):
    """
    Removes repeated lines from the generated content.
    
    Args:
        gen_content (str): The raw generated content from the model.
    
    Returns:
        str: Cleaned content with unique lines.
    """
    lines = gen_content.splitlines()
    unique_lines = []
    for line in lines:
        if line not in unique_lines:
            unique_lines.append(line)
    return "\n".join(unique_lines)

# Function to extract classification result
def extract_classification_result(cleaned_content):
    """
    Extracts the classification result from the cleaned content and removes trailing asterisks, if any.
    
    Args:
        cleaned_content (str): The cleaned content from the model.
    
    Returns:
        str: The classification stage or "Not Found" if not present.
    """
    if "Classification Stage" in cleaned_content:
        try:
            # Extract the part after "Classification Stage"
            classification = cleaned_content.split("Classification Stage")[-1].split()[0].strip()
            # Remove any trailing asterisks
            return classification.rstrip('*')
        except IndexError:
            return "Not Found"
    return "Not Found"


def prepare_initial_prompt(template_text, role="system"):
    """
    Prepare the initial system prompt for generation or reflection.

    Args:
        template_text (str): The default text template for the task.
        role (str): The role for the message (default is "system").

    Returns:
        dict: A message dictionary for the chat history.
    """
    return {
        "role": role,
        "content": template_text
    }

def prepare_prompt_with_history(chat_history, new_content, role="user"):
    """
    Add a new message to the chat history.

    Args:
        chat_history (list): The existing chat history.
        new_content (str): The new message content.
        role (str): The role for the new message (default is "user").

    Returns:
        list: Updated chat history with the new message appended.
    """
    chat_history.append({
        "role": role,
        "content": new_content
    })
    return chat_history

def perform_generation(processor, model, chat_history, image, max_new_tokens=1024):
    """
    Perform the generation step using chat history.

    Args:
        processor: The processor for preparing inputs.
        model: The model used for generation.
        chat_history (list): The chat history for generation.
        image: The image input for the model.
        max_new_tokens (int): The maximum number of tokens to generate.

    Returns:
        str: The cleaned content generated by the model.
    """
    inputs = prepare_input(processor, chat_history, image)
    gen_content = generate_output(model, processor, inputs, max_new_tokens)
    cleaned_content = clean_generated_content(gen_content)
    return cleaned_content

def perform_reflection(processor, model, chat_history, image, max_new_tokens=1024):
    """
    Perform the reflection step using chat history.

    Args:
        processor: The processor for preparing inputs.
        model: The model used for reflection.
        chat_history (list): The chat history for reflection.
        image: The image input for the model.
        max_new_tokens (int): The maximum number of tokens to generate.

    Returns:
        str: The critique generated by the model.
    """
    inputs = prepare_input(processor, chat_history, image)
    critique = generate_output(model, processor, inputs, max_new_tokens)
    return critique


def prepare_initial_prompt_with_image(template_text, role="system"):
    """
    Prepare the initial system prompt for generation or reflection, including an image.

    Args:
        template_text (str): The default text template for the task.
        image: The image to include in the initial prompt.
        role (str): The role for the message (default is "system").

    Returns:
        dict: A message dictionary for the chat history.
    """
    msg = [ 
        {
        "role": role,
        "content": [
            {"type": "text", "text": template_text},
            {"type": "image"}
        ]
    }
    ]
    return msg


def iterative_generation_and_reflection(processor, model, prompt_method, template_generation, template_reflection, user_prompt, image, num_iterations=2, max_new_tokens=1024):
    """
    Perform iterative generation and reflection steps with evolving chat history.

    Args:
        processor: The processor for preparing inputs.
        model: The model used for generation and reflection.
        template_generation (str): The default system template for generation.
        template_reflection (str): The default system template for reflection.
        image: The image input for the model.
        num_iterations (int): Number of iterations to perform.
        max_new_tokens (int): The maximum number of tokens to generate.

    Returns:
        None
    """
    # Initialize chat histories with image included
    generation_chat_history = prepare_initial_prompt_with_image(template_generation, role="user") # this is equivalent to msg in the LLama code 
    reflection_chat_history = prepare_initial_prompt_with_image(template_reflection, role="user")

    for iteration in range(1, num_iterations + 1):
        print(f"\n\n\n\n==========Iteration:{iteration}========\n\n\n\n")

        print("\n\n\nStarting Generation Block...\n\n\n")

        # check_gpu_memory()
        # print('\n\n\n')

        # add user prompt to gneeration history
        generation_chat_history = prepare_prompt_with_history(
            generation_chat_history, user_prompt, role="user"
        )


        # Step 1: Perform Generation
        generation_result = perform_generation(
            processor, model, generation_chat_history, image, max_new_tokens
        )

        # print(f"\n-----------\nGenerated LLM Output: ")
        # display_markdown(generation_result)

        # extract assigned class 
        

        # Add generation result to histories
        generation_chat_history = prepare_prompt_with_history(
            generation_chat_history, generation_result, role="user"
        )
        reflection_chat_history = prepare_prompt_with_history(
            reflection_chat_history, generation_result, role="user"
        )

        # Step 2: Perform Reflection
        print("\n\n\nStarting Reflection Block...\n\n\n")
        # check_gpu_memory()
        # print('\n\n\n')

        reflection_result = perform_reflection(
            processor, model, reflection_chat_history, image, max_new_tokens
        )

        print("\n-----------------\n Generated Critique:")
        display_markdown(reflection_result)

        # Add reflection result to histories
        reflection_chat_history = prepare_prompt_with_history(
            reflection_chat_history, reflection_result, role="critique"
        )
        generation_chat_history = prepare_prompt_with_history(
            generation_chat_history, reflection_result, role="critique"
        )
