import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from datasets import DatasetDict
import os
import json 
from PIL import Image

torch.set_default_device("cuda")
logging.set_verbosity_error()

# Create model
model = AutoModelForCausalLM.from_pretrained(
    "models/imp-v1-3b/",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("models/imp-v1-3b/", trust_remote_code=True)


rsvqa_lr = DatasetDict.load_from_disk(os.path.join("data", "rsvqa-lr", "hf"))
rsvqa_hr = DatasetDict.load_from_disk(os.path.join("data", "rsvqa-hr", "hf"))

with open(os.path.join("data", "prompts.json"), "r") as f:
    prompts = json.load(f)


def build_prompt(question: str, question_type: str) -> str:
    system_message = "You are an artificial intelligence assistant that helps users with questions concerning orbital imagery based on a set of possible answers."
    user_message = prompts[question_type].replace("<question>", f"{question}{'?' if not question.endswith('?') else ''}")
    return "\n".join([system_message, user_message])

def ask(prompt: str, image: Image) -> str:
    # Set inputs
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    image_tensor = model.image_preprocess(image)

    # Generate the answer
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        images=image_tensor,
        use_cache=True)[0]

    return tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

# eval rsvqa-lr
for split in ["test"]:
    for row in rsvqa_lr[split]:
        prompt = build_prompt(row["question"], row["type"])
        answer = ask(prompt, row["img"])
        row["img"].save("test.png")
        print(f"{prompt}{answer}")
        print(f"Correct answer:{row['answer']}")
        exit()

        


