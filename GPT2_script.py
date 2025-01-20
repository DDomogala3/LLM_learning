from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example usage with improved parameters
input_text = input("")
inputs = tokenizer(input_text, return_tensors="pt")

# Generate response with controlled parameters
outputs = model.generate(
    **inputs,
    max_length=50,          # Maximum length of the generated sequence
    num_return_sequences=1, # Number of response sequences to generate
    temperature=0.7,        # Sampling temperature
    top_k=50,               # Top-k sampling
    top_p=0.9,              # Nucleus sampling
    do_sample=True,         # Enable sampling
    pad_token_id=tokenizer.eos_token_id  # Set pad_token_id to eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
