# Seam Carving Augmentation + Siamese Network

A project that applies **Seam Carving-based data augmentation** to the Hymenoptera (ants/bees) binary classification dataset and trains a **Siamese Network** to measure visual similarity between two images.

---

## Screenshots

| Different class — Ant vs Bee (62.8%) | Same class — Bee vs Bee (99.7%) | File dialog — Ant vs Ant (99.7%) |
|:---:|:---:|:---:|
| ![Ant vs Bee](ScrShot%206.png) | ![Bee vs Bee](ScrShot%208.png) | ![Ant vs Ant](ScrShot%2010.png) |
| Red bar: dissimilar prediction | Green bar: similar prediction | Right image loaded from local PC |

---

## Objectives

- Use Seam Carving augmentation to remove and vary backgrounds, guiding the model to focus on **the object itself**
- Train a Siamese Network so that **same-class pairs (bee–bee, ant–ant)** yield high similarity and **different-class pairs (bee–ant)** yield low similarity
- Test the trained model in real time through a GUI application

---

## File Structure

```
.
├── hymenoptera/                 # Dataset
│   ├── train/
│   │   ├── ants/  (1,230 images)
│   │   └── bees/  (1,210 images)
│   └── val/
│       ├── ants/  (700 images)
│       └── bees/  (830 images)
│
├── checkpoints/
│   ├── best_siamese.pth         # Best Val AUC model (Epoch 5)
│   ├── last_siamese.pth         # Final epoch model (Epoch 25)
│   └── history.npy              # Training curves (loss / acc / auc per epoch)
│
├── seam_carving_augment.py      # Seam Carving augmentation script
├── siamese_train.py             # Siamese Network training script
├── siamese_gui.py               # Similarity measurement GUI app
└── README.md
```

---

## Pipeline

### Step 1 — Data Preprocessing

Two preprocessing steps applied to the original 397 images:

| Step | Details |
|------|---------|
| Remove corrupt image | Delete `train/ants/imageNotFound.gif` |
| Unify resolution | Resize shorter side to 256, then CenterCrop to **224×224** |

```bash
# Preprocessing has already been applied; no need to re-run
```

### Step 2 — Seam Carving Data Augmentation

Generate 9 variants per original image using content-aware resizing → **10× augmentation**

| Tag | Seams Removed | Description |
|-----|--------------|-------------|
| `sc_w05` | 11 horizontal | Width reduced by 5%, restored to 224×224 |
| `sc_w10` | 22 horizontal | Width reduced by 10% |
| `sc_w20` | 44 horizontal | Width reduced by 20% |
| `sc_h05` | 11 vertical | Height reduced by 5% |
| `sc_h10` | 22 vertical | Height reduced by 10% |
| `sc_h20` | 44 vertical | Height reduced by 20% |
| `sc_wh05` | 11 each direction | Both axes reduced by 5% |
| `sc_wh10` | 22 each direction | Both axes reduced by 10% |
| `sc_wh20` | 44 each direction | Both axes reduced by 20% |

Low-energy pixels (background) are removed first, so the subject is preserved while the background composition changes.

```
397 original images → 3,970 augmented images  (train 2,440 / val 1,530)
```

```bash
python3 seam_carving_augment.py
```

### Step 3 — Siamese Network Training

**Architecture**

```
Image A ──┐
          ├── [Shared weights] ResNet18 backbone
Image B ──┘   └── avgpool → FC(512→256) → LayerNorm → ReLU → Dropout → FC(256→128) → L2 normalize
                             ↓                                                              ↓
                        embedding A                                                   embedding B
                             └──────────── cosine similarity ∈ [-1, 1] ────────────────────┘
```

**Loss Function — Contrastive Loss (cosine distance)**

```
dist = 1 - cosine_sim          # ∈ [0, 2]

L = label × dist²  +  (1 - label) × max(0, margin - dist)²
       ↑ same class: minimize distance      ↑ different class: push distance > margin
```

| Hyperparameter | Value |
|---------------|-------|
| Backbone | ResNet18 (ImageNet pretrained) |
| Embedding dim | 128 |
| Margin | 0.5 |
| Optimizer | Adam (backbone lr=5e-5, head lr=3e-4) |
| Scheduler | CosineAnnealingLR |
| Batch size | 64 pairs |
| Epochs | 25 |
| Train pairs/epoch | 1,500 |
| Val pairs/epoch | 500 |

**Training Results**

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

### Step 4 — Run the GUI App

Load the trained model and measure similarity between two images in real time.  
**Windows 2000 Classic** style interface.

```bash
python3 siamese_gui.py
```

**Features**
- Left panel: `>> Load Random Image` button — picks a random image from the dataset
- Right panel: `>> Load from File...` button — opens an OS file dialog to load any image from your PC
- Segmented progress bar (green = similar / red = dissimilar)
- Bottom status bar shows raw cosine similarity value

---

## Inference Example

```python
from siamese_train import SiameseNet
import torch, torchvision.transforms as T
from PIL import Image

# Load model
model = SiameseNet(embed_dim=128)
ckpt  = torch.load("checkpoints/best_siamese.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

# Preprocessing
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
print(f"cosine sim = {sim:.4f}  →  {'same class' if sim > 0.5 else 'different class'}")
```

---

## Dependencies

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

## Design Intent

Seam Carving removes **low-energy pixels (background) first** based on an energy map.  
As a result, augmented images have varying background compositions while the subject (ant/bee) remains intact.  
Training the Siamese Network with these background-varied pairs as positive examples encourages the embedding space to become **background-invariant**,  
so the model learns to measure similarity based on the object rather than its surroundings.
