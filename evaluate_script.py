import json
import os
import subprocess
import shutil
import tarfile

model_tar_path = "/opt/ml/processing/model/model.tar.gz"
extract_dir = "/opt/ml/processing/model"


with tarfile.open(model_tar_path) as tar:
    tar.extractall(path=extract_dir)
    
print("File trong model dir:", os.listdir(extract_dir))


subprocess.run(['git', 'clone', 'https://github.com/ttmq-2423/medical_mae.git'], check=True)
os.chdir('medical_mae')

# Run the evaluation script
try: 
    result = subprocess.run([
        "python", "Brute_force.py", 
        "--batch_size", "8",
        "--finetune", "/opt/ml/processing/model/checkpoint.pth",
        "--model", "conv_vit",
        "--data_path", "data/CheXpert-v1.0/",  
        "--num_workers", "1",
        "--train_list", "data/CheXpert-v1.0/train.csv", 
        "--val_list", "data/CheXpert-v1.0/test1.csv", 
        "--test_list", "data/CheXpert-v1.0/test1.csv", 
        "--nb_classes", "5",
        "--dataset", "chexpert",
        "--aa", "rand-m6-mstd0.5-inc1",
        "--device", "cpu",
        "--save", "figure"
    ], check=True, capture_output=True, text=True)  # important: capture output as string
except subprocess.CalledProcessError as e:
    print("Command failed:")
    print("STDOUT:\n", e.stdout)
    print("STDERR:\n", e.stderr)
    raise

destination_dir = '/opt/ml/processing/evaluation' 
os.makedirs(destination_dir, exist_ok=True)

# Save stdout and stderr to result.txt
with open(os.path.join(destination_dir, "result.txt"), "w") as f:
   f.write(result.stdout)
   f.write(result.stderr)

print(result.stdout)
print(result.stderr)

# Extract metrics
auc_avg = 0.0
auc_per_label = []

for line in result.stdout.splitlines():
    if "AUC avg:" in line:
        try:
            auc_avg = float(line.split("AUC avg:")[1].split("%")[0].strip())
        except Exception as e:
            print(f"Error parsing AUC avg: {e}")
    elif "AUC for each label:" in line:
        try:
            label_str = line.split("AUC for each label:")[1].strip()
            label_str = label_str.strip("[]")
            auc_per_label = [float(x.strip()) for x in label_str.split(",")]
        except Exception as e:
            print(f"Error parsing AUC per label: {e}")
    elif "Optimal thresholds per class:" in line:
        try:
            thresh_str = line.split("Optimal thresholds per class:")[1].strip().strip("[]")
            optimal_thresholds = [float(x.strip()) for x in thresh_str.split(",")]
        except Exception as e:
            print(f"Error parsing optimal thresholds: {e}")

print(f"Extracted AUC avg: {auc_avg}")
print(f"Extracted AUC per label: {auc_per_label}")
print(f"Extracted Optimal thresholds: {optimal_thresholds}")



class_names = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']

report_dict = {
    "metrics": {
        "auc_avg": {
            "value": float(auc_avg)
        }
    },
    "visualizations": [
        {
            "name": "confusion_matrix",
            "value": "confusion_matrix_conv_vit.png",
            "content_type": "image/png"
        },
        {
            "name": "roc_curve", 
            "value": "ROC_curves_conv_vit_improved.png",
            "content_type": "image/png"
        }
    ]
}

for i, auc in enumerate(auc_per_label):
    report_dict["metrics"][f"auc_{class_names[i].replace(' ', '_').lower()}"] = {
        "value": float(auc)
    }

for i, thresh in enumerate(optimal_thresholds):
    class_name_key = class_names[i].replace(" ", "_").lower()
    report_dict["metrics"][f"optimal_threshold_{class_name_key}"] = {
        "value": float(thresh)
    }

# Write to JSON
with open(os.path.join(destination_dir, "evaluation.json"), "w") as f:
    json.dump(report_dict, f, indent=4)

log_dir = './figure/'
shutil.copytree(log_dir, destination_dir, dirs_exist_ok=True)

print(f"Metrics report created successfully to {destination_dir}")
