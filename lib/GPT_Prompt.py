import csv
import os
import gradio as gr

# GPT Prompt
PROMPTS_CSV_PATH = "saved_prompts.csv"

def get_prompts_from_csv():
    if not os.path.exists(PROMPTS_CSV_PATH):
        return []
    with open(PROMPTS_CSV_PATH, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        # remove empty rows
        return [row[0] for row in reader if row]
            
def save_prompt(prompt):
    # Append CSV
    with open(PROMPTS_CSV_PATH, 'a+', newline='', encoding='utf-8') as file:
        # Move to start
        file.seek(0)
        reader = csv.reader(file)
        existing_prompts = [row[0] for row in reader]
        if prompt not in existing_prompts:
            writer = csv.writer(file)
            writer.writerow([prompt])
        # Move to end
        file.seek(0, os.SEEK_END)
    return gr.Dropdown(label="Saved Prompts", choices=get_prompts_from_csv(), type="value", interactive=True)

def delete_prompt(prompt):
    lines = []
    with open(PROMPTS_CSV_PATH, 'r', newline='', encoding='utf-8') as readFile:
        reader = csv.reader(readFile)
        lines = [row for row in reader if row and row[0] != prompt]
    with open(PROMPTS_CSV_PATH, 'w', newline='', encoding='utf-8') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)
    return gr.Dropdown(label="Saved Prompts", choices=get_prompts_from_csv(), type="value", interactive=True)