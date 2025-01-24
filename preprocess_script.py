import subprocess
import tarfile
import os




# Chuyển đến thư mục chứa các tệp cần thiết
os.chdir('/opt/ml/processing/input')
subprocess.run("pip install -r requirements_processing.txt", shell=True, check=True)


# Chạy script Python xử lý dữ liệu
subprocess.run([
    "python", "processing.py",
    "--input_size", "224",
    "--random_resize_range", "0.5", "1.0",
    "--datasets_names", "chexpert"
], check=True)

# Nén file dataset_train.pkl
input_file = 'dataset_train.pkl'
output_file = '/opt/ml/processing/output/data_train.tar.gz'

with tarfile.open(output_file, 'w:gz') as tar:
    tar.add(input_file)

# Tải lên tệp nén lên S3 nếu cần
# subprocess.run(f"aws s3 cp {output_file} s3://{default_s3_bucket}", shell=True, check=True)

print("Processing completed and output uploaded to S3.")
