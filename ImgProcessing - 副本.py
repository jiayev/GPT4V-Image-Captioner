import os
import concurrent.futures
from PIL import Image
from tqdm import tqdm

target_resolutions = [
    (704, 1472),
    (768, 1360),
    (832, 1248),
    (864, 1184),
    (1024, 1024),
    (1184, 864),
    (1248, 832),
    (1360, 768),
    (1072, 704),
]

def convert_image_to_jpg(img, img_path):
    """Convert an Image object to JPG."""
    # Remove extension from original filename and add .jpg
    base_name = os.path.splitext(img_path)[0]
    jpg_path = base_name + '.jpg'
    
    # Convert image to RGB if it is RGBA (or any other mode)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img.save(jpg_path, format='JPEG', quality=100)

def process_image(img_path):
    try:
        if img_path.lower().endswith((".jpg", ".png", ".bmp", ".gif", ".tif",".tiff", ".jpeg",".webp")):  # 添加或删除需要的图像文件类型
            img = Image.open(img_path)

            # 计算原图像的宽高比
            original_aspect_ratio = img.width / img.height

            # 找到最接近原图像宽高比的目标分辨率
            target_resolution = min(target_resolutions, key=lambda res: abs(original_aspect_ratio - res[0] / res[1]))

            # 计算新的维度
            if img.width / target_resolution[0] < img.height / target_resolution[1]:
                new_width = target_resolution[0]
                new_height = int(img.height * target_resolution[0] / img.width)
            else:
                new_height = target_resolution[1]
                new_width = int(img.width * target_resolution[1] / img.height)

            # 等比缩放图像
            img = img.resize((new_width, new_height), Image.LANCZOS)

            # 计算裁剪的区域
            left = int((img.width - target_resolution[0]) / 2)
            top = int((img.height - target_resolution[1]) / 2)
            right = int((img.width + target_resolution[0]) / 2)
            bottom = int((img.height + target_resolution[1]) / 2)

            # 裁剪图像
            img = img.crop((left, top, right, bottom))

            # 转换并保存图像为JPG格式
            convert_image_to_jpg(img, img_path)
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None  # Or handle the error as needed

def delete_non_jpg_files(folder_path):
    """Delete all non-jpg image files in a directory, but keep txt files."""
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if not filename.lower().endswith((".jpg", ".txt")):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error occurred while deleting file : {file_path}. Error : {str(e)}")

def process_images_in_folder(folder_path):
    """
    Process all images in the given folder according to the target resolutions,
    then delete all non-jpg files except for .txt files.
    """
    file_list = [os.path.join(dirpath, filename)
                 for dirpath, dirnames, filenames in os.walk(folder_path)
                 for filename in filenames]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_image, file_list), total=len(file_list)))
    
    delete_non_jpg_files(folder_path)
    return f"Processed images in folder: {folder_path}"