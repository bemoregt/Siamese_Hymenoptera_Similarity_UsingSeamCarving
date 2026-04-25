"""
Siamese Network 학습 — 벌/개미 유사도 측정
backbone: ResNet18 (pretrained) + L2-normalized projection head
loss:     Contrastive Loss (cosine distance 기반)
"""

import os, glob, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from sklearn.metrics import roc_auc_score

# ── 설정 ────────────────────────────────────────────────────────────────────

CFG = dict(
    root       = "hymenoptera",
    embed_dim  = 128,
    margin     = 0.5,          # contrastive loss margin (cosine dist 기반)
    n_train    = 1500,         # epoch당 학습 pair 수
    n_val      = 500,          # epoch당 검증 pair 수
    batch_size = 64,
    epochs     = 25,
    lr_backbone= 5e-5,
    lr_head    = 3e-4,
    weight_decay=1e-4,
    seed       = 42,
    save_dir   = "checkpoints",
    threshold  = 0.5,          # 분류 임계값 (cosine sim > threshold → 동일 클래스)
)

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
os.makedirs(CFG["save_dir"], exist_ok=True)


# ── Dataset ─────────────────────────────────────────────────────────────────

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# 정규화된 float32 텐서를 미리 캐시해 transform 오버헤드 제거
# 학습 시 RandomHorizontalFlip만 온-더-플라이 적용
_TO_TENSOR  = T.ToTensor()
_NORMALIZE  = T.Normalize(_MEAN, _STD)


class PairDataset(Dataset):
    """이미지를 정규화된 float32 텐서로 메모리에 사전 캐싱"""

    def __init__(self, root, split, n_pairs, augment=False, seed=0):
        self.n_pairs = n_pairs
        self.augment = augment   # train=True → RandomHorizontalFlip
        self.samples = {}

        print(f"  [{split}] 텐서 사전 계산 중 ...", end="", flush=True)
        self.cache = {}
        for cls in ["ants", "bees"]:
            path = os.path.join(root, split, cls)
            files = sorted(glob.glob(os.path.join(path, "*.jpg")))
            self.samples[cls] = files
            for f in files:
                img = Image.open(f).convert("RGB")
                self.cache[f] = _NORMALIZE(_TO_TENSOR(img))   # (3,224,224) float32
        print(f" {len(self.cache)}개 완료")

        self.classes = list(self.samples.keys())
        self.pairs   = self._gen_pairs(random.Random(seed))

    def _gen_pairs(self, rng):
        pairs = []
        for i in range(self.n_pairs):
            if i % 2 == 0:
                cls  = rng.choice(self.classes)
                a, b = rng.sample(self.samples[cls], 2)
                pairs.append((a, b, 1.0))
            else:
                c1, c2 = rng.sample(self.classes, 2)
                a = rng.choice(self.samples[c1])
                b = rng.choice(self.samples[c2])
                pairs.append((a, b, 0.0))
        return pairs

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        t1 = self.cache[p1].clone()
        t2 = self.cache[p2].clone()
        if self.augment:
            if random.random() < 0.5:
                t1 = torch.flip(t1, [2])   # horizontal flip
            if random.random() < 0.5:
                t2 = torch.flip(t2, [2])
        return t1, t2, torch.tensor(label, dtype=torch.float32)


# ── Model ────────────────────────────────────────────────────────────────────

class SiameseNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # conv1 ~ avgpool (children 0..8), 마지막 FC 제거
        self.backbone = nn.Sequential(*list(base.children())[:-1])   # → (B, 512, 1, 1)
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
        )

    def embed(self, x):
        feat = self.backbone(x)              # (B, 512, 1, 1)
        emb  = self.projector(feat)          # (B, embed_dim)
        return F.normalize(emb, dim=1)       # L2 정규화 → cosine sim = 내적

    def forward(self, x1, x2):
        e1 = self.embed(x1)
        e2 = self.embed(x2)
        sim = (e1 * e2).sum(dim=1)           # cosine similarity ∈ [-1, 1]
        return sim


# ── Loss ─────────────────────────────────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    """
    label=1 → 같은 클래스 → sim 높이기 (dist 줄이기)
    label=0 → 다른 클래스 → sim 낮추기 (dist > margin)
    dist = 1 - cosine_sim  ∈ [0, 2]
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, sim, labels):
        dist = 1.0 - sim                                      # [0, 2]
        pos  = labels       * dist.pow(2)
        neg  = (1 - labels) * F.relu(self.margin - dist).pow(2)
        return 0.5 * (pos + neg).mean()


# ── 학습 / 평가 루프 ──────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, all_sim, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for img1, img2, labels in loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            sim  = model(img1, img2)
            loss = criterion(sim, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_sim.append(sim.detach().cpu())
            all_labels.append(labels.cpu())

    all_sim    = torch.cat(all_sim).numpy()
    all_labels = torch.cat(all_labels).numpy().astype(int)

    avg_loss = total_loss / len(loader)
    acc      = ((all_sim > CFG["threshold"]) == all_labels).mean()
    auc      = roc_auc_score(all_labels, all_sim)

    return avg_loss, acc, auc


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print(f"device: {device}")
    print(f"embed_dim={CFG['embed_dim']}, margin={CFG['margin']}, "
          f"epochs={CFG['epochs']}, batch={CFG['batch_size']}\n")

    # Dataset / Loader
    train_ds = PairDataset(CFG["root"], "train", CFG["n_train"], augment=True,  seed=CFG["seed"])
    val_ds   = PairDataset(CFG["root"], "val",   CFG["n_val"],   augment=False, seed=CFG["seed"]+1)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=0)

    # Model / Loss / Optimizer
    model     = SiameseNet(embed_dim=CFG["embed_dim"]).to(device)
    criterion = ContrastiveLoss(margin=CFG["margin"])
    optimizer = optim.Adam([
        {"params": model.backbone.parameters(),  "lr": CFG["lr_backbone"]},
        {"params": model.projector.parameters(), "lr": CFG["lr_head"]},
    ], weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"], eta_min=1e-6
    )

    best_val_auc = 0.0
    history = []

    print(f"{'Ep':>3}  {'TrainLoss':>9} {'TrainAcc':>8} {'TrainAUC':>8}  "
          f"{'ValLoss':>8} {'ValAcc':>7} {'ValAUC':>7}  {'Time':>5}")
    print("-" * 75)

    for ep in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        # 매 epoch마다 새로운 pair 생성
        train_ds.pairs = train_ds._gen_pairs(random.Random(ep * 100))

        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc, va_auc = run_epoch(model, val_loader,   criterion, None,      device, train=False)
        scheduler.step()

        elapsed = time.time() - t0
        history.append(dict(ep=ep, tr_loss=tr_loss, tr_acc=tr_acc, tr_auc=tr_auc,
                            va_loss=va_loss, va_acc=va_acc, va_auc=va_auc))

        flag = ""
        if va_auc > best_val_auc:
            best_val_auc = va_auc
            torch.save({"epoch": ep, "model": model.state_dict(),
                        "cfg": CFG, "val_auc": va_auc},
                       os.path.join(CFG["save_dir"], "best_siamese.pth"))
            flag = " ★"

        print(f"{ep:>3}  {tr_loss:>9.4f} {tr_acc:>7.3f}  {tr_auc:>7.4f}  "
              f"{va_loss:>8.4f} {va_acc:>7.3f} {va_auc:>7.4f}  {elapsed:>4.0f}s{flag}")

    # 마지막 체크포인트 저장
    torch.save({"epoch": CFG["epochs"], "model": model.state_dict(), "cfg": CFG},
               os.path.join(CFG["save_dir"], "last_siamese.pth"))

    # 학습 곡선 저장
    np.save(os.path.join(CFG["save_dir"], "history.npy"), history)

    print(f"\n학습 완료 — Best Val AUC: {best_val_auc:.4f}")
    print(f"모델 저장: {CFG['save_dir']}/best_siamese.pth")


if __name__ == "__main__":
    main()
