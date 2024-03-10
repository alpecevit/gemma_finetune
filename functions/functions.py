

# function to generate the input text
def generate_inputs(row, na_list):
    prefix = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n'
    if row['input'] and row['input'] not in na_list:
        text = f"<start_of_turn>user {prefix} {row['instruction']} using inputs {row['input']} <end_of_turn>\n<start_of_turn>model {row['output']}<end_of_turn>"
    else:
        text = f"<start_of_turn>user {prefix} {row['instruction']}<end_of_turn> \n<start_of_turn>model {row['output']}<end_of_turn>"
    return text