import importlib
import subprocess

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
        if libs.find("==") == -1: #only fix requirements which contain "==", because I don't know how to take "<=",">="... into account at the same time.
            check_libs = libs
        else:
            check_libs = libs[:libs.index("==")]
        if check_libs == "Pillow": #import PIL instead of import Pillow
            check_libs = "PIL"
        try:
            importlib.import_module(check_libs)
        except ImportError:
            missing_libs.append(check_libs)

    return missing_libs

def print_missing(missing_libs):
    # 返
    if missing_libs == []:
        return ""
    else:
        return f"Not installed libraries: {', '.join(missing_libs)}"

# 启动检查
def check_open():
    require_path = "./install_script/requirements.txt"
    missings = install_detection(require_path)
    installed = print_missing(missings)
    if installed == "":
        return
    else:
        print(installed)
        for lib in missings:
            subprocess.check_call(["pip", "install", lib])

if __name__ == "__main__":
    check_open()
