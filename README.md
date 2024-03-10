# Finetune Gemma

This project is about fine-tuning the Gemma model from Google using the Hugging Face Transformers library.

## Dependencies

- Hugging Face Transformers
- Datasets
- Torch
- Accelerate
- PEFT
- TRL

## Base Model

The base model being fine-tuned is [google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it).

## Dataset

The dataset used for fine-tuning is [iamtarun/code_instructions_120k_alpaca](https://huggingface.co/datasets/iamtarun/code_instructions_120k_alpaca).

## Training

Training was done on Google Colab using T4 GPU.

## Usage

The code from `prompting.py` can be used directly on trained model. Or, the following chunk can be used on trained model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gemma-fintune-code")
tokenizer = AutoTokenizer.from_pretrained("gemma-fintune-code")

input_text = "write me a python code to calculate factorial of a number"
input_ids = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)
outputs = model.generate(**input_ids, max_new_tokens=300,
                                 pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Example Output
```python
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

print(factorial(5))
```

**Output:**

```python
120
```

**Explanation:**

* The function uses recursion to calculate the factorial of a number n.
* The base case is when n is 0, which returns 1.
* For all other values of n, it recursively calls itself with n-1 and multiplies the result by n.
* The time complexity of this algorithm is O(n), where n is the input number.
* The space complexity is O(1), as we only need to store the current and previous factorials.

## Example Output From Base Model
```python
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)


print(factorial(5))
```

**Output:**

```python
120
```

**Explanation:**

* The `factorial()` function takes a single integer argument, `n`.
* It uses recursion to calculate the factorial of `n`.
* If `n` is 0, it returns 1 (the factorial of 0 is defined as 1).
* Otherwise, it returns `n` multiplied by the factorial of `n-1`.
* The function continues this recursive process until `n` reaches 0, at which point it starts returning the results of previous recursive calls.
* The `print()` statement calls the `factorial()` function with the argument `5`, which triggers the recursive calculation and prints the result.
