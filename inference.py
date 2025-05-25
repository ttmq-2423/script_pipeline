from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import os
import boto3
import json

# === Custom model ===
import torch.nn as nn
import timm

class TransformerEncoder(nn.Module):
    def __init__(self, dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x):
        x_res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + x_res

        x_res = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + x_res
        return x

class MobileViTBlock(nn.Module):
    def __init__(self, dim, ffn_dim, n_transformer_blocks=2):
        super().__init__()
        self.local_rep = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.transformer = nn.Sequential(
            *[TransformerEncoder(dim, ffn_dim) for _ in range(n_transformer_blocks)]
        )

    def forward(self, x):
        x = self.local_rep(x)
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_trans = self.transformer(x_flat)
        x_trans = x_trans.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + x_trans  # residual connection

class ConvNeXtMobileViT(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model("convnext_tiny", features_only=True, pretrained=False)
        self.mobilevit = MobileViTBlock(dim=384, ffn_dim=768, n_transformer_blocks=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(384, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        x = feats[2]  # Use Stage 2
        x = self.mobilevit(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x  # for multi-label classification

# === Flask App ===
app = Flask(__name__)
class_names = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']

# Hàm tải thresholds từ S3
def load_thresholds_from_s3():
    s3_client = boto3.client('s3')
    
    # Lấy S3 URI từ biến môi trường
    eval_json_s3_uri = os.environ.get('EVALUATION_JSON_S3_URI')
    if not eval_json_s3_uri:
        raise ValueError("EVALUATION_JSON_S3_URI environment variable not set!")
    
    # Parse bucket và key từ S3 URI
    bucket = eval_json_s3_uri.split('/')[2]
    key = '/'.join(eval_json_s3_uri.split('/')[3:])
    
    # Tải file từ S3
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        evaluation_data = json.loads(response['Body'].read().decode('utf-8'))
        
        # Trích xuất thresholds từ evaluation data
        thresholds = {
            'Cardiomegaly': evaluation_data['metrics']['optimal_threshold_cardiomegaly']['value'],
            'Edema': evaluation_data['metrics']['optimal_threshold_edema']['value'],
            'Consolidation': evaluation_data['metrics']['optimal_threshold_consolidation']['value'],
            'Atelectasis': evaluation_data['metrics']['optimal_threshold_atelectasis']['value'],
            'Pleural Effusion': evaluation_data['metrics']['optimal_threshold_pleural_effusion']['value']
        }
        return thresholds
    except Exception as e:
        raise RuntimeError(f"Failed to load thresholds from S3: {str(e)}")

# Load model và thresholds
def load_model():
    # Load model
    model = ConvNeXtMobileViT(num_classes=len(class_names))
    checkpoint = torch.load('/opt/ml/model/checkpoint.pth', map_location='cpu')
    
    if 'state_dict' in checkpoint:
        checkpoint_model = checkpoint['state_dict']
    elif 'model' in checkpoint:
        checkpoint_model = checkpoint['model']
    else:
        checkpoint_model = checkpoint

    model.load_state_dict(checkpoint_model, strict=False)
    model.eval()
    model.to(torch.device('cpu'))
    
    # Load thresholds
    thresholds = load_thresholds_from_s3()
    
    return model, thresholds

model, thresholds = load_model()

# Tiền xử lý ảnh
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Health check
@app.route("/ping", methods=["GET"])
def ping():
    return "OK", 200

# Dự đoán với áp dụng thresholds
@app.route("/invocations", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "image" not in data:
            return jsonify({"error": "Missing 'image' key in JSON request"}), 400

        image_bytes = base64.b64decode(data["image"])
        image_tensor = preprocess_image(image_bytes)

        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.sigmoid(logits).tolist()[0]  # sigmoid for multi-label

        # Áp dụng thresholds và xác định bệnh
        results = []
        for i, class_name in enumerate(class_names):
            prob = probabilities[i]
            threshold = thresholds[class_name]
            if prob >= threshold:
                results.append(class_name)

        # Trả về danh sách các bệnh được phát hiện
        return jsonify({
            "detected_diseases": results,
            "message": "Found diseases" if results else "No diseases detected"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
