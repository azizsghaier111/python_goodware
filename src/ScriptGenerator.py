import ast
import os
import openai
import json
import random

# Constants
NB_LINES = '100'
MAX_SCRIPTS = 30
PYTHON_VERSION = '3.9'
CODES_DIR = 'codes2'
RECORD_FILE_PATH = 'record.txt'
API_KEY = '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'

# Initialize OpenAI client
client1 = openai.OpenAI(api_key=API_KEY)


# Function to extract functionalities from the response
def extract_functionalities(string):
    s = str(string)
    start_index = s.find("[")
    end_index = s.find("]")

    if start_index != -1 and end_index != -1:
        list_string = s[start_index:end_index + 1]
        return list_string
    else:
        return -1


# Function to extract random elements from a list
def extract_random_elements(lst, n=3):
    if len(lst) < n:
        raise ValueError("List must contain at least as many elements as n")
    return random.sample(lst, n)


# Function to extract code from the response
def extract_code(string):
    start_index = string.find("```python")
    if start_index == -1:
        start_index = string.find("```Python")
    if start_index == -1:
        return string

    end_index = string.find("```", start_index + 9)
    if end_index == -1:
        return string

    code = string[start_index + 9:end_index].strip()
    return code


# Function to load the library collisions JSON file
def load_library_collisions(filename):
    with open(filename, 'r') as file:
        return json.load(file)


# Function to create directories if they don't exist
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Function to read the current progress from the record file
def read_progress(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()
            current_lib_index = int(lines[0].strip())
            current_script_index = int(lines[1].strip())
            return current_lib_index, current_script_index
    else:
        return 0, 0


# Function to write progress to the record file
def write_progress(filepath, lib_index, script_index):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    lines[0] = f"{lib_index}\n"
    lines[1] = f"{script_index}\n"
    with open(filepath, 'w') as file:
        file.writelines(lines)


# Function to prompt for library functionalities
def get_library_functionalities(lib):
    while True:
        functionalities_prompt = f"what are the top 20 main functionalities of {lib}? Put them in a Python list (example: [functionality1, functionality2, functionality3...])."
        chat_completion = client1.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": functionalities_prompt,
                }
            ],
            model="gpt-4",
        )
        content = chat_completion.choices[0].message.content
        lib_functionalities = extract_functionalities(content)
        if lib_functionalities != -1:
            lib_functionalities = ast.literal_eval(lib_functionalities)
            if len(lib_functionalities) > 2:
                return lib_functionalities


# Function to generate and save scripts for a library
def generate_scripts_for_library(lib_index, lib, common_libs, lib_functionalities, lib_dir, start_script_index):
    code_list = []
    for script_index in range(start_script_index, MAX_SCRIPTS):
        funcs = extract_random_elements(lib_functionalities)
        if not code_list:
            script_prompt = f"develop a Python {PYTHON_VERSION} script that uses the {lib} library in order to ensure {funcs} along with {', '.join(common_libs)} if necessary, please try always to include the main. Please only the script! At least {NB_LINES} lines."
        else:
            script_prompt = f"develop a Python {PYTHON_VERSION} script that uses the {lib} library in order to ensure {funcs} along with {', '.join(common_libs)} if necessary, please try always to include the main. Try to merge it if possible with {random.choice(code_list)}. Please only the script! At least {NB_LINES} lines."

        chat_completion = client1.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": script_prompt
                }
            ],
            model="gpt-4",
        )
        content = chat_completion.choices[0].message.content
        code = extract_code(content)

        code_list.append(code)
        script_path = os.path.join(lib_dir, f"{script_index}.py")
        with open(script_path, "w") as file:
            file.write(code)

        write_progress(RECORD_FILE_PATH, lib_index, (script_index + 1) % MAX_SCRIPTS)


def main():
    create_directory(CODES_DIR)
    library_collisions = load_library_collisions('library_collisions.json')
    current_lib_index, current_script_index = read_progress(RECORD_FILE_PATH)
    keys = list(library_collisions.keys())

    for lib_index in range(current_lib_index, len(keys)):
        lib = keys[lib_index]
        print(f"Processing library: {lib}")
        common_libs = library_collisions[lib]
        lib_dir = os.path.join(CODES_DIR, lib)
        create_directory(lib_dir)

        lib_functionalities = get_library_functionalities(lib)
        write_progress(RECORD_FILE_PATH, lib_index, 0)
        generate_scripts_for_library(lib_index, lib, common_libs, lib_functionalities, lib_dir, current_script_index)

    print("Scripts saved to 'codes' directory.")


if __name__ == "__main__":
    main()

