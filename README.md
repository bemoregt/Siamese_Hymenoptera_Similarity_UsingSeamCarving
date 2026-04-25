# Seam Carving Augmentation + Siamese Network

Hymenoptera(개미/벌) 이진 분류 데이터셋에 **Seam Carving 기반 데이터 증강**을 적용하고,  
증강된 데이터로 **Siamese Network**를 학습하여 두 이미지 간 시각적 유사도를 측정하는 프로젝트.

---

## 스크린샷

| 다른 종류 — 개미 vs 벌 (62.8%) | 같은 종류 — 벌 vs 벌 (99.7%) | 파일 불러오기 — 개미 vs 개미 (99.7%) |
|:---:|:---:|:---:|
| ![개미 vs 벌](ScrShot%206.png) | ![벌 vs 벌](ScrShot%208.png) | ![개미 vs 개미](ScrShot%2010.png) |
| 빨간 바: 비유사 판정 | 초록 바: 유사 판정 | 우측 이미지를 PC에서 직접 로딩 |

---

## 프로젝트 목표

- 배경을 제거·변형하는 Seam Carving 증강으로 모델이 **물체 자체**에 집중하도록 유도
- Siamese Network가 **같은 클래스(벌-벌, 개미-개미)** → 높은 유사도, **다른 클래스(벌-개미)** → 낮은 유사도를 출력하도록 학습
- 학습된 모델을 GUI 앱에서 실시간으로 테스트

---

## 파일 구조

```
.
├── hymenoptera/                 # 데이터셋
│   ├── train/
│   │   ├── ants/  (1,230장)
│   │   └── bees/  (1,210장)
│   └── val/
│       ├── ants/  (700장)
│       └── bees/  (830장)
│
├── checkpoints/
│   ├── best_siamese.pth         # Val AUC 최고 모델 (Epoch 5)
│   ├── last_siamese.pth         # 마지막 epoch 모델 (Epoch 25)
│   └── history.npy              # 학습 곡선 (loss / acc / auc per epoch)
│
├── seam_carving_augment.py      # Seam Carving 증강 스크립트
├── siamese_train.py             # Siamese Network 학습 스크립트
├── siamese_gui.py               # 유사도 측정 GUI 앱
└── README.md
```

---

## 파이프라인

### 1단계 — 데이터 전처리

원본 데이터셋 397장에 대해 두 가지 전처리를 적용:

| 처리 | 내용 |
|------|------|
| 불량 이미지 제거 | `train/ants/imageNotFound.gif` 삭제 |
| 해상도 통일 | 짧은 변 → 256 리사이즈 후 CenterCrop **224×224** |

```bash
# 전처리는 이미 적용된 상태이므로 별도 실행 불필요
```

### 2단계 — Seam Carving 데이터 증강

콘텐츠 인식 리사이징(Seam Carving)으로 원본 1장당 9가지 변형을 생성 → **10배 증강**

| 태그 | 제거 시임 수 | 설명 |
|------|-------------|------|
| `sc_w05` | 가로 11 | 폭 5% 압축 후 224×224 복원 |
| `sc_w10` | 가로 22 | 폭 10% 압축 |
| `sc_w20` | 가로 44 | 폭 20% 압축 |
| `sc_h05` | 세로 11 | 높이 5% 압축 |
| `sc_h10` | 세로 22 | 높이 10% 압축 |
| `sc_h20` | 세로 44 | 높이 20% 압축 |
| `sc_wh05` | 가로+세로 각 11 | 양방향 5% |
| `sc_wh10` | 가로+세로 각 22 | 양방향 10% |
| `sc_wh20` | 가로+세로 각 44 | 양방향 20% |

에너지가 낮은 픽셀(배경)을 우선 제거하므로 피사체는 보존되고 배경 구성만 달라짐.

```
원본 397장 → 증강 후 3,970장  (train 2,440 / val 1,530)
```

```bash
python3 seam_carving_augment.py
```

### 3단계 — Siamese Network 학습

**구조**

```
입력 이미지 A ──┐
                ├── [공유 가중치] ResNet18 backbone
입력 이미지 B ──┘   └── avgpool → FC(512→256) → LayerNorm → ReLU → Dropout → FC(256→128) → L2 normalize
                                   ↓                                                              ↓
                              embedding A                                                   embedding B
                                   └──────────── cosine similarity ∈ [-1, 1] ────────────────────┘
```

**손실 함수 — Contrastive Loss (cosine distance 기반)**

```
dist = 1 - cosine_sim          # ∈ [0, 2]

L = label × dist²  +  (1 - label) × max(0, margin - dist)²
       ↑ 같은 클래스: 거리 최소화        ↑ 다른 클래스: 거리 > margin 유지
```

| 하이퍼파라미터 | 값 |
|---------------|-----|
| Backbone | ResNet18 (ImageNet pretrained) |
| Embedding dim | 128 |
| Margin | 0.5 |
| Optimizer | Adam (backbone lr=5e-5, head lr=3e-4) |
| Scheduler | CosineAnnealingLR |
| Batch size | 64 pairs |
| Epochs | 25 |
| Train pairs/epoch | 1,500 |
| Val pairs/epoch | 500 |

**학습 결과**

| Epoch | Train Loss | Train AUC | Val Loss | Val AUC |
|------:|----------:|----------:|--------:|--------:|
| 1 | 0.0278 | 0.689 | 0.0135 | 0.935 |
| 2 | 0.0060 | 0.997 | 0.0131 | 0.943 |
| 3 | 0.0016 | 1.000 | 0.0133 | 0.946 |
| **5** | **0.0007** | **1.000** | **0.0120** | **0.955 ★** |
| 25 | 0.0001 | 1.000 | 0.0129 | 0.949 |

> Best Val AUC **0.9548** (Epoch 5) — `checkpoints/best_siamese.pth`

```bash
python3 siamese_train.py
```

### 4단계 — GUI 앱 실행

학습된 모델을 로딩하여 두 이미지의 유사도를 실시간 측정.  
**Windows 2000 Classic** 스타일 인터페이스.

```bash
python3 siamese_gui.py
```

**기능**
- 이미지 1 / 이미지 2 패널 각각에 `>> 랜덤 이미지 불러오기` 버튼
- 버튼 클릭 시 데이터셋에서 무작위 이미지 로딩 + 즉시 유사도 재계산
- 세그먼트형 프로그레스 바 (초록 = 유사 / 빨강 = 비유사)
- 하단 상태바에 raw cosine similarity 값 표시

---

## 모델 추론 예시

```python
from siamese_train import SiameseNet
import torch, torchvision.transforms as T
from PIL import Image

# 모델 로딩
model = SiameseNet(embed_dim=128)
ckpt  = torch.load("checkpoints/best_siamese.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

# 전처리
tf = T.Compose([
    T.Resize(224), T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def similarity(path1, path2):
    def embed(p):
        t = tf(Image.open(p).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            return model.embed(t)
    sim = (embed(path1) * embed(path2)).sum().item()   # cosine sim ∈ [-1, 1]
    return sim

sim = similarity("img_bee.jpg", "img_ant.jpg")
print(f"cosine sim = {sim:.4f}  →  {'같은 종류' if sim > 0.5 else '다른 종류'}")
```

---

## 의존 패키지

```
torch >= 2.0
torchvision
Pillow
numpy
scikit-learn
```

```bash
pip install torch torchvision Pillow numpy scikit-learn
```

---

## 핵심 설계 의도

Seam Carving은 에너지 맵 기반으로 **배경의 저에너지 픽셀부터 제거**한다.  
따라서 증강된 이미지들은 배경 구성이 다르지만 피사체(개미/벌)의 형태는 그대로 유지된다.  
이 다양한 배경 변형 쌍을 Siamese Network의 positive pair로 학습시키면,  
모델의 임베딩 공간이 **배경 불변(background-invariant)** 특성을 갖도록 유도된다.
