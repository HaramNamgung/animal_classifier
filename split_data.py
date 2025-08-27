#split_dataset.py
import argparse, random, shutil
from pathlib import Path

def main(src="data", dst="data_split", val_ratio=0.2):
    src = Path(src)
    dst_train = Path(dst) / "train"
    dst_val   = Path(dst) / "val"
    dst_train.mkdir(parents=True, exist_ok=True)
    dst_val.mkdir(parents=True, exist_ok=True)

    # image type 
    exts = {".jpg",".jpeg",".png",".webp",".bmp",".gif"}

    classes = [d for d in src.iterdir() if d.is_dir()]
    print(f"[class number] {len(classes)}")
    for c in sorted(classes, key=lambda p: p.name):
        files = [p for p in c.rglob("*") if p.suffix.lower() in exts]
        random.shuffle(files)
        n_val = max(1, int(len(files)*val_ratio))
        val_files = set(files[:n_val])
        tr_files = [p for p in files if p not in val_files]

        # Make new folders in a data_split folder 
        (dst_train/c.name).mkdir(parents=True, exist_ok=True)
        (dst_val/c.name).mkdir(parents=True, exist_ok=True)

        # copy (원본은 data/에)
        for p in tr_files:
            shutil.copy2(p, dst_train/c.name/p.name)
        for p in val_files:
            shutil.copy2(p, dst_val/c.name/p.name)

        print(f"- {c.name}: train : {len(tr_files)} // val : {len(val_files)}")

    print("split complete!")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data", help="원본 데이터가 있는 폴더 (클래스별 하위폴더 포함)를 입력하시오.")
    ap.add_argument("--dst", default="data_split", help="나눠서 복사될 목적지 폴더를 입력하시오.")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="검증 비율 (예: 0.2 = 20%) 입력하시오.")
    args = ap.parse_args()
    main(args.src, args.dst, args.val_ratio)
