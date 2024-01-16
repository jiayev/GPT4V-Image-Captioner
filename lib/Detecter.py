import importlib
import GPUtil

def check_memory():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        if gpu.memoryTotal > 12000:
            return ""
    return "Insufficient GPU graphics memory for use. / 显存不足"

def install_detection():
    # 读
    file_path = "./install_script/check.txt"
    requirements = []
    with open(file_path, 'r') as file:
        for line in file:
            requirements.append(line.strip())

    # 查
    missing_libs = []
    for libs in requirements:
        try:
            importlib.import_module(libs)
        except ImportError:
            missing_libs.append(libs)
    
    # 返
    if missing_libs == []:
        output = ""
    else:
        output = f"Not installed libraries: {', '.join(missing_libs)}"
    return output

def detecter():
    gpu_check = check_memory()
    if gpu_check == "":
        installed = install_detection()
        if installed == "":
            return "All listed libraries are installed. / Cog安装无误"
        else:
            return installed
    else:
        return gpu_check
    
def is_installed(package):
    try:
        dist = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        try:
            spec = importlib.util.find_spec(package)
        except ModuleNotFoundError:
            return False

        return spec is not None

    return dist is not None