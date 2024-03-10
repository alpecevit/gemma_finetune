from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gemma-fintune-code")
tokenizer = AutoTokenizer.from_pretrained("gemma-fintune-code")

input_text = "write me a code to solve knapsack problem in python"
input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)
outputs = model.generate(**input_ids, max_new_tokens=300,
                                 pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))