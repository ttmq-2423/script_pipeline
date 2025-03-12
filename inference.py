from flask import Flask, request, jsonify
import torch
import torchvision.models as models
import os
import random

app = Flask(__name__)

# Định nghĩa danh sách class
class_names = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

seed_everything(10)
device = torch.device('cpu')

# Hàm load model
def load_model():
    model_instance = models.__dict__['densenet121'](num_classes=len(class_names))

    checkpoint = torch.load('/opt/ml/model/Pretrain_densenet121.pth', map_location=torch.device('cpu'))
    print('type checkpoint.pth:', type(checkpoint))
    print('Checkpoint keys:', checkpoint.keys())

    if 'state_dict' in checkpoint:
        checkpoint_model = checkpoint['state_dict']
    elif 'model' in checkpoint:
        checkpoint_model = checkpoint['model']
    else:
        checkpoint_model = checkpoint

    model_instance.load_state_dict(checkpoint_model, strict=False)
    model_instance.eval()
    model_instance.to(device)

    return model_instance

# Load model khi container khởi động
model = load_model()

@app.route("/ping", methods=["GET"])
def ping():
    return "OK", 200  # Health check

@app.route("/invocations", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_tensor = torch.tensor(data["input"]).to(device)

        with torch.no_grad():
            output = model(input_tensor).tolist()

        return jsonify({"predictions": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
