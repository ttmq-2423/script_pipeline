import tarfile
import shutil
import subprocess
import os

subprocess.run(['git', 'clone', 'https://github.com/ttmq-2423/medical_mae.git'], check=True)
os.chdir('medical_mae')

subprocess.run([
    'python', 'main_finetune_chestxray.py',
    '--output_dir', './OUTPUT/',
    '--log_dir', './LOG/',
    '--batch_size', '8',
    '--input_size', '224',
    '--epochs', '1',
    '--blr', '2.5e-4',
    '--weight_decay', '0.05',
    '--model', 'conv_vit',
    '--warmup_epochs', '5',
    '--drop_path', '0',
    '--mixup', '0',
    '--cutmix', '0',
    '--reprob', '0',
    '--vit_dropout_rate', '0',
    '--data_path', 'data/CheXpert-v1.0/',
    '--num_workers', '1',
    '--train_list', 'data/CheXpert-v1.0/train.csv',
    '--test_list', 'data/CheXpert-v1.0/test1.csv',
    '--nb_classes', '5',
    '--eval_interval', '1',
    '--min_lr', '1e-5',
    '--dataset', 'chexpert',
    '--build_timm_transform',
    '--aa', 'rand-m6-mstd0.5-inc1',
    '--device', 'cpu'
], check=True)

log_dir = './LOG/'
destination_dir = '/opt/ml/output/'
os.makedirs(destination_dir, exist_ok=True)
shutil.copytree(log_dir, destination_dir, dirs_exist_ok=True)

log_dir = './OUTPUT/log.txt'
destination_dir = '/opt/ml/output/'
os.makedirs(destination_dir, exist_ok=True)
shutil.copy(log_dir, destination_dir, dirs_exist_ok=True)


model_dir = './OUTPUT/checkpoint.pth'
destination_dir = '/opt/ml/model/'

os.makedirs(destination_dir, exist_ok=True)

shutil.copy(model_dir, destination_dir, dirs_exist_ok=True)
