from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

# Danh sách class bệnh
class_names = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']


def load_model():
    import torchvision.models as models
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
    model_instance.to(torch.device('cpu'))

    return model_instance

# Load model khi container khởi động
model = load_model()

# Tiền xử lý ảnh
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

@app.route("/ping", methods=["GET"])
def ping():
    return "OK", 200  # Health check

@app.route("/invocations", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image_tensor = preprocess_image(file.read())

        with torch.no_grad():
            output = model(image_tensor).tolist()

        return jsonify({"predictions": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
