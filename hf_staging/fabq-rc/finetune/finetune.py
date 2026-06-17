from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model on CPU (no GPU available)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu"
)

# Test prompt
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))