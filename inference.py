from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

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

# Load model
def load_model():
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
    return model

model = load_model()

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

# Dự đoán
@app.route("/invocations", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "image" not in data:
            return jsonify({"error": "Missing 'image' key in JSON request"}), 400

        image_bytes = base64.b64decode(data["image"])
        image_tensor = preprocess_image(image_bytes)

        with torch.no_grad():
            output = model(image_tensor).sigmoid().tolist()  # sigmoid for multi-label

        predictions = {class_names[i]: output[0][i] for i in range(len(class_names))}
        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
