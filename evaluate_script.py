import json
import os
import subprocess
import shutil
import tarfile

# Get the source code directory
source_dir = '/opt/ml/input/data/code'  
destination_dir = '/opt/ml/code'  # Destination directory

# Check if the destination directory exists, and create it if it doesn't
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Loop through the source directory and copy files to the destination directory
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # Full path to the source file
        source_file = os.path.join(root, file)
        
        # Create the destination path, preserving subdirectory structure
        relative_path = os.path.relpath(root, source_dir)  # Compute the relative path from the source
        destination_folder = os.path.join(destination_dir, relative_path)  # Create the destination folder to copy into

        # Check if destination folder exists and make it if it doesn't
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Full path to the destination file
        destination_file = os.path.join(destination_folder, file)

        # Copy the file to the new directory
        shutil.copy(source_file, destination_file)

        print(f'Successfully copied: {source_file} -> {destination_file}')


# Path to save model artifacts
model_output_dir = '/opt/ml/model' # use this dir to store model artifacts

# Run the evaluation script
result = subprocess.run([
    "python", "evaluate.py",
    "--batch_size", "8",
    "--finetune", "Pretrain_densenet121.pth",
    "--model", "densenet121",
    "--data_path", "/opt/ml/input/data/training/",
    "--num_workers", "11",
    "--test_list", "/opt/ml/input/data/training/test1.csv",
    "--nb_classes", "5",
    "--eval_interval", "10",
    "--dataset", "chexpert",
    "--aa", "rand-m6-mstd0.5-inc1",
    "--device", "cpu"
], capture_output=True, text=True, check=True)

# Capture the output from evaluate.py and save into result.txt
with open(os.path.join(model_output_dir, "result.txt"), "w") as f:
   f.write(result.stdout)
   f.write(result.stderr)
print(result.stdout)
print(result.stderr)

# Extract auc from the result
auc_avg = 0 # default value if can't get
for line in result.stdout.splitlines():
  if "AUC avg: " in line:
    auc_avg = float(line.split("AUC avg: ")[1].split()[0])

print(f"Extracted auc value: {auc_avg}")

# Create the metrics report dictionary
report_dict = {
    "regression_metrics": {
        "auc": {
            "value": auc_avg,
             }
    }
}

# Write the report to a file, in the model_output_dir
with open(os.path.join(model_output_dir, "evaluation.json"), "w") as f:
    json.dump(report_dict, f)

print(f"Metrics report created successfully to {model_output_dir}")