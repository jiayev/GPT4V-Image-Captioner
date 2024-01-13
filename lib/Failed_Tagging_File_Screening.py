import argparse
import os
import shutil

# List of supported image file extensions
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif']

# Global variable to keep track of the number of moved images
moved_images_count = 0

# Check and move documents and their associated image files with the same name
def move_files_with_keywords(source_folder, target_folder, keywords):
    global moved_images_count
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Iterate through all the files and folders within source_folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                # Check if the text file contains any of the keywords
                if has_keywords(file_path, keywords):
                    # Move the text file
                    shutil.move(file_path, os.path.join(target_folder, file))
                    # Move the related image files with the same name
                    move_related_images(root, file, target_folder)

# Check if the text file contains any of the given keywords
def has_keywords(file_path, keywords):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return any(keyword.lower() in content.lower() for keyword in keywords)

# Move the image files with the same name as the text file
def move_related_images(file_dir, text_file, target_folder):
    global moved_images_count
    base_name = os.path.splitext(text_file)[0]
    for ext in IMAGE_EXTENSIONS:
        image_file = base_name + ext
        image_path = os.path.join(file_dir, image_file)
        if os.path.exists(image_path):
            shutil.move(image_path, os.path.join(target_folder, image_file))
            moved_images_count += 1  # Increment the count for each moved image

def main(image_path, keywords):
    # The target folder will be created in the same directory as image_path
    target_folder = os.path.join(os.path.dirname(image_path), 'moved_files')
    move_files_with_keywords(image_path, target_folder, keywords)
    # Display the message with the count of moved images and the target folder path
    print(f"Operation complete / 操作完成. Total images moved: {moved_images_count}. Moved to folder: {target_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move documents containing keywords and their associated image files with the same name.")
    parser.add_argument('--image_path', type=str, help='The path to the folder')
    parser.add_argument('--keywords', type=str, help='List of keywords, separated by commas', default='error,sorry,content')
    args = parser.parse_args()
    
    # Split the received keyword string into a list
    keywords = args.keywords.split(',')
    
    main(args.image_path, keywords)