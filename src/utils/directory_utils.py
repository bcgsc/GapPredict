import os

def get_terminal_directory_character():
    return "\\" if os.name == 'nt' else "/"

def clean_directory_string(dir):
    clean_dir = str(dir)
    terminal_directory_character = get_terminal_directory_character()
    if clean_dir[-1] != terminal_directory_character:
        clean_dir += terminal_directory_character
    return clean_dir

def mkdir(path):
    os.makedirs(path, exist_ok=True)