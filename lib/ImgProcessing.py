import os
import concurrent.futures
from PIL import Image
from tqdm import tqdm
from PIL import Image, ExifTags

target_resolutions = [
    (640, 1632),   # 640 * 1632 = 1044480
    (704, 1472),   # 704 * 1472 = 1036288
    (768, 1360),   # 768 * 1360 = 1044480
    (832, 1248),   # 832 * 1248 = 1038336
    (896, 1152),
    (960, 1088),   # 960 * 1088 = 1044480
    (992, 1056),   # 992 * 1056 = 1047552
    (1024, 1024),  # 1024 * 1024 = 1048576
    (1056, 992),   # 1056 * 992 = 1047552
    (1088, 960),   # 1088 * 960 = 1044480
    (1152, 896),
    (1248, 832),   # 1248 * 832 = 1038336
    (1360, 768),   # 1360 * 768 = 1044480
    (1472, 704),   # 1472 * 704 = 1036288
    (1632, 640),   # 1632 * 640 = 1044480
    # (768, 1360),   # 768 * 1360 = 1044480
    # (1472, 704),   # 1472 * 704 = 1036288
    # (1024, 1024),  # 1024 * 1024 = 1048576
]

# This function will rotate the image according to its EXIF orientation
def apply_exif_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()

        if exif is not None:
            exif = dict(exif.items())
            orientation_value = exif.get(orientation)

            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError, TypeError):
        # cases: image don't have getexif
        pass

    return image

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
        if img_path.lower().endswith((".jpg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".jpeg", ".webp")):
            img = Image.open(img_path)
            img = apply_exif_orientation(img)  # Apply the EXIF orientation

            # Convert to 'RGB' if it is 'RGBA' or any other mode
            img = img.convert('RGB')

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
        return None  # Consider more detailed error handling logic

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
    processed_files = []

    file_list = [os.path.join(dirpath, filename)
                 for dirpath, dirnames, filenames in os.walk(folder_path)
                 for filename in filenames]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_image, file_list), total=len(file_list)))

    delete_non_jpg_files(folder_path)
    return f"Processed images in folder: {folder_path}"