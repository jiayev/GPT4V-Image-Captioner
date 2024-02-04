import importlib
import GPUtil

def check_memory():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        if gpu.memoryTotal > 12000:
            return ""
    return "Insufficient GPU graphics memory for use. / 显存过小"

def install_detection(requir_path):
    # 读
    file_path = requir_path
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

    return missing_libs

def print_missing(missing_libs):
    # 返
    if missing_libs == []:
        return ""
    else:
        return f"Not installed libraries: {', '.join(missing_libs)}"

def detecter():
    gpu_check = check_memory()
    if gpu_check == "":
        cog_requir = "./install_script/check.txt"
        installed = print_missing(install_detection(cog_requir))

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
