# predict_fixed.py
from pathlib import Path
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CKPT = 'models_simple/best.pt'
CLASSES_TXT = 'models_simple/classes.txt'
IMG_PATH = r"C:\Users\82105\animal_classifier\predict\my_file.webp"  # <- 여기만 네 파일로!

# classes 로드
ckpt = torch.load(CKPT, map_location='cpu')
classes = ckpt.get('classes') or Path(CLASSES_TXT).read_text(encoding='utf-8').splitlines()

# 모델 만들고 가중치 로드
try:
    model = models.resnet18(weights=None)
except:
    model = models.resnet18(pretrained=False)
in_feat = model.fc.in_features
model.fc = nn.Linear(in_feat, len(classes))
model.load_state_dict(ckpt['model_state'])
model.eval()

# 전처리 + 예측
tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
im = Image.open(IMG_PATH).convert('RGB')
x = tf(im).unsqueeze(0)

with torch.inference_mode():
    probs = torch.softmax(model(x), dim=1)[0]
top = probs.topk(5)
print(f"파일: {Path(IMG_PATH).name}")
for p, idx in zip(top.values.tolist(), top.indices.tolist()):
    print(f"- {classes[idx]}: {p:.3f}")
