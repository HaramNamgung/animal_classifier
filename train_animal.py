import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------------------
# 1) 옵션
# ---------------------

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_split', help='train/val 폴더가 들어있는 루트')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--out', default='models_simple')
    ap.add_argument('--cpu', action='store_true', help='강제로 CPU 사용')
    ap.add_argument('--pretrained', action='store_true', help='ImageNet 가중치 사용(인터넷 필요)')
    return ap.parse_args()

# ---------------------
# 2) 전처리/데이터로더
# ---------------------

def make_loaders(root: str, img_size: int, batch_size: int, device: torch.device):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dir = Path(root) / 'train'
    val_dir   = Path(root) / 'val'

    train_ds = datasets.ImageFolder(train_dir, transform=tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=tf)

    pin = (device.type == 'cuda')

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=pin)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=pin)

    return train_ld, val_ld, train_ds.classes

# ---------------------
# 3) 모델
# ---------------------

def build_model(num_classes: int, lr: float, use_pretrained: bool):
    # torchvision 버전 차이 안전 처리
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=use_pretrained)

    # 마지막 레이어 교체(클래스 수에 맞게)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    crit = nn.CrossEntropyLoss()
    return model, opt, crit

# ---------------------
# 4) 학습/검증 루프 (아주 단순)
# ---------------------

def train_one_epoch(model, loader, crit, opt, device):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for i, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += y.size(0)

        if i % 20 == 0 or i == len(loader):
            print(f"  - train batch {i}/{len(loader)}")

    return total_loss / max(1, total_n), total_correct / max(1, total_n)


def eval_one_epoch(model, loader, crit, device):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.inference_mode():
        for i, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n += y.size(0)
            if i % 50 == 0 or i == len(loader):
                print(f"  - val batch {i}/{len(loader)}")
    return total_loss / max(1, total_n), total_correct / max(1, total_n)

# ---------------------
# 5) 메인
# ---------------------

def main():
    args = get_args()
    device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print('[Device]', device)

    train_ld, val_ld, classes = make_loaders(args.data, args.img_size, args.batch_size, device)
    print('[Classes]', len(classes), '종 →', classes)

    model, opt, crit = build_model(num_classes=len(classes), lr=args.lr, use_pretrained=args.pretrained)
    model.to(device)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'classes.txt').write_text('\n'.join(classes), encoding='utf-8')

    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        print(f"\n[Epoch {ep}/{args.epochs}]")
        tr_loss, tr_acc = train_one_epoch(model, train_ld, crit, opt, device)
        va_loss, va_acc = eval_one_epoch(model, val_ld, crit, device)
        print(f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

        # last 저장
        torch.save({'model_state': model.state_dict(), 'classes': classes}, out / 'last.pt')
        # best 갱신
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({'model_state': model.state_dict(), 'classes': classes}, out / 'best.pt')
            print(f"  ✅ saved best.pt (val_acc={best_acc:.3f})")

    print('\nDone. Weights @', out / 'best.pt')

if __name__ == '__main__':
    main()
