import os
import subprocess
import re

def find_python_pyc_files(directory):
    """
    Recursively finds all Python bytecode files (.pyc) in the specified directory.

    Args:
    - directory (str): Directory to search for .pyc files.

    Returns:
    - list: List of paths to .pyc files found.
    """
    python_pyc_files = []
    pattern = re.compile(r'\d+\.pyc$')  # Pattern to match .pyc files
    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern.match(file):
                python_pyc_files.append(os.path.join(root, file))
    return python_pyc_files

def decompile_pyc_file(pyc_file, output_directory):
    """
    Decompiles a single .pyc file using pycdc.

    Args:
    - pyc_file (str): Path to the .pyc file to decompile.
    - output_directory (str): Directory where the decompiled .py file will be saved.
    """
    try:
        # Determine relative path from input directory
        rel_path = os.path.relpath(pyc_file, input_directory)
        output_dir = os.path.join(output_directory, os.path.dirname(rel_path))
        output_dir = os.path.abspath(output_dir)

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Determine the name of the decompiled file (without extension)
        decompiled_filename = os.path.splitext(os.path.basename(pyc_file))[0] + ".py"
        decompiled_file_path = os.path.join(output_dir, decompiled_filename)

        print(f'Decompiling {pyc_file} to {decompiled_file_path}...')

        # Run pycdc to decompile the .pyc file
        subprocess.check_call(["pycdc", pyc_file, "-o", decompiled_file_path])
        print(f"Successfully decompiled {pyc_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to decompile {pyc_file}. Error message: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def decompile_pyc_files(input_directory, output_directory):
    """
    Main function to decompile all .pyc files in a directory.

    Args:
    - input_directory (str): Directory containing .pyc files to decompile.
    - output_directory (str): Directory where decompiled .py files will be saved.
    """
    python_pyc_files = find_python_pyc_files(input_directory)

    for pyc_file in python_pyc_files:
        decompile_pyc_file(pyc_file, output_directory)

# Specify the input directory containing .pyc files
input_directory = 'pyc_files'

# Specify the output directory where decompiled .py files will be saved
output_directory = 'decompiled_files'

# Call the function to decompile .pyc files
decompile_pyc_files(input_directory, output_directory)
