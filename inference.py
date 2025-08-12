import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_name = "mistralai/Mistral-7B-v0.1"
lora_model_path = "./lora_finetuned_model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval()

def generate_answer(question, max_length=256):
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        user_input = input("Customer Question: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = generate_answer(user_input)
        print(f"Bot Answer: {response}")
