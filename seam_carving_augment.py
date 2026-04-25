"""
Seam Carving Data Augmentation
각 이미지에 대해 9가지 시임 카빙 변형을 생성하여 10배 증강
"""

import numpy as np
from PIL import Image
import os
import glob
from multiprocessing import Pool, cpu_count
from functools import partial

# ── Seam Carving 핵심 함수 ──────────────────────────────────────────────────

def compute_energy(img):
    """듀얼 그라디언트 에너지 함수 (e1)"""
    gray = img @ np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    # 양방향 차분으로 경계 artifact 방지
    dy = np.abs(np.roll(gray, -1, axis=0) - np.roll(gray, 1, axis=0))
    dx = np.abs(np.roll(gray, -1, axis=1) - np.roll(gray, 1, axis=1))
    return (dx + dy).astype(np.float32)


def find_vertical_seam(energy):
    """DP로 최소 에너지 수직 시임 탐색"""
    h, w = energy.shape
    M = energy.copy()
    back = np.zeros((h, w), dtype=np.int8)

    for i in range(1, h):
        prev = M[i - 1]
        left   = np.empty(w, dtype=np.float32); left[0]  = np.inf; left[1:]  = prev[:-1]
        right  = np.empty(w, dtype=np.float32); right[-1] = np.inf; right[:-1] = prev[1:]
        stacked = np.stack([left, prev, right])          # (3, w)
        idx     = np.argmin(stacked, axis=0)             # 0/1/2
        back[i] = idx - 1                                # -1/0/+1
        M[i]   += stacked[idx, np.arange(w)]

    seam = np.empty(h, dtype=np.int32)
    seam[-1] = int(np.argmin(M[-1]))
    for i in range(h - 2, -1, -1):
        seam[i] = int(np.clip(seam[i + 1] + back[i + 1, seam[i + 1]], 0, w - 1))
    return seam


def remove_vertical_seam(img, seam):
    """수직 시임 1개 제거"""
    h, w, c = img.shape
    mask = np.ones((h, w), dtype=bool)
    mask[np.arange(h), seam] = False
    return img[mask].reshape(h, w - 1, c)


def carve_width(img, n):
    """수직 시임 n개 제거 (폭 축소)"""
    for _ in range(n):
        seam = find_vertical_seam(compute_energy(img.astype(np.float32)))
        img  = remove_vertical_seam(img, seam)
    return img


def carve_height(img, n):
    """수평 시임 n개 제거 (높이 축소) — 전치 활용"""
    img_t = np.ascontiguousarray(np.transpose(img, (1, 0, 2)))
    img_t = carve_width(img_t, n)
    return np.ascontiguousarray(np.transpose(img_t, (1, 0, 2)))


def seam_carve(img, n_w, n_h):
    """폭·높이 각각 n_w, n_h 시임 제거 후 224×224 복원"""
    if n_w > 0:
        img = carve_width(img, n_w)
    if n_h > 0:
        img = carve_height(img, n_h)
    # 224×224로 리사이즈 복원
    return np.array(Image.fromarray(img).resize((224, 224), Image.LANCZOS))


# ── 증강 변형 정의 ──────────────────────────────────────────────────────────
# (tag, width_seams, height_seams)  — 224의 5/10/20%
VARIANTS = [
    ("sc_w05",  11,  0),
    ("sc_w10",  22,  0),
    ("sc_w20",  44,  0),
    ("sc_h05",   0, 11),
    ("sc_h10",   0, 22),
    ("sc_h20",   0, 44),
    ("sc_wh05", 11, 11),
    ("sc_wh10", 22, 22),
    ("sc_wh20", 44, 44),
]

BASE = "/Users/m1_4k/0415_qnn/0425_SeamCarving_resnet/hymenoptera"
SIZE = 224


# ── 단일 이미지 처리 ────────────────────────────────────────────────────────

def process_image(img_path):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    # 이미 증강된 파일이면 건너뜀
    if any(v[0] in stem for v in VARIANTS):
        return 0

    img = np.array(Image.open(img_path).convert("RGB"))
    out_dir = os.path.dirname(img_path)
    saved = 0

    for tag, n_w, n_h in VARIANTS:
        save_path = os.path.join(out_dir, f"{stem}_{tag}.jpg")
        if os.path.exists(save_path):
            continue
        result = seam_carve(img.copy(), n_w, n_h)
        Image.fromarray(result).save(save_path, "JPEG", quality=95)
        saved += 1

    return saved


# ── 메인 ────────────────────────────────────────────────────────────────────

def main():
    all_files = []
    for split in ["train", "val"]:
        for cls in ["ants", "bees"]:
            path = os.path.join(BASE, split, cls)
            files = sorted(glob.glob(os.path.join(path, "*.jpg")))
            # 원본 파일만 (증강 파일 제외)
            originals = [f for f in files
                         if not any(v[0] in os.path.basename(f) for v in VARIANTS)]
            all_files.extend(originals)

    total_orig = len(all_files)
    print(f"원본 이미지: {total_orig}개")
    print(f"생성 예정:   {total_orig * len(VARIANTS)}개 (총 {total_orig * (len(VARIANTS)+1)}개)")
    print(f"CPU 코어:    {cpu_count()}개 사용\n")

    n_workers = min(cpu_count(), 8)
    done = 0
    with Pool(n_workers) as pool:
        for i, saved in enumerate(pool.imap_unordered(process_image, all_files), 1):
            done += saved
            pct = i / total_orig * 100
            print(f"\r  진행: {i}/{total_orig} ({pct:.1f}%)  저장됨: {done}개", end="", flush=True)

    print(f"\n\n완료: 증강 이미지 {done}개 생성")

    # 결과 요약
    print("\n=== 최종 데이터셋 크기 ===")
    for split in ["train", "val"]:
        for cls in ["ants", "bees"]:
            path = os.path.join(BASE, split, cls)
            n = len(glob.glob(os.path.join(path, "*.jpg")))
            print(f"  {split}/{cls}: {n}개")


if __name__ == "__main__":
    main()
