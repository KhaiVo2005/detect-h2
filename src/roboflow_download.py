import os
import shutil
from roboflow import Roboflow
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    rf = Roboflow(api_key="0wcPLEtbOE65BLx9HipH")
    project = rf.workspace("only-me-h4rdp").project("h2detect-vrdju")
    version = project.version(5)
    dataset = version.download("yolov8")
    
    downloaded_dir = dataset.location

    target_dir = os.path.join("datasets", "h2detect-vrdju")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.move(downloaded_dir, target_dir)

    os.makedirs("data", exist_ok=True)
    yaml_src = os.path.join(target_dir, "data.yaml")
    yaml_dst = os.path.join("data", "data.yaml")
    shutil.copy(yaml_src, yaml_dst)
    
    import os
import shutil
from roboflow import Roboflow
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    rf = Roboflow(api_key="0wcPLEtbOE65BLx9HipH")
    project = rf.workspace("only-me-h4rdp").project("h2detect-vrdju")
    version = project.version(5)
    dataset = version.download("yolov8")
    
    downloaded_dir = dataset.location

    target_dir = os.path.join("datasets", "h2detect-vrdju")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.move(downloaded_dir, target_dir)

    os.makedirs("data", exist_ok=True)
    yaml_src = os.path.join(target_dir, "data.yaml")
    yaml_dst = os.path.join("data", "data.yaml")
    shutil.copy(yaml_src, yaml_dst)

    with open(yaml_dst, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith("train:"):
            new_lines.append("train: ../datasets/h2detect-vrdju/train/images\n")
        elif line.startswith("val:"):
            new_lines.append("val: ../datasets/h2detect-vrdju/valid/images\n")
        elif line.startswith("test:"):
            new_lines.append("test: ../datasets/h2detect-vrdju/test/images\n")
        else:
            new_lines.append(line)

    with open(yaml_dst, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    for file_name in ["README.dataset.txt", "README.roboflow.txt", "data.yaml"]:
        file_path = os.path.join(target_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

    print("Dataset and data.yaml have been successfully downloaded and organized.")

if __name__ == "__main__":
    main()
                