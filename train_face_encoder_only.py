"""
train_face_encoder_only.py  —  Pretrain FaceEncoder as emotion+intensity classifier

Trains the FaceEncoder CNN from scratch on MEAD face images using:
  - emotion classification head  (6 classes: angry/fear/happy/surprised/sad/neutral)
  - intensity classification head (3 classes: level1/level2/level3)
    using OrdinalLoss for intensity (level2 is hardest — ordinal aware)

After training, only the FaceEncoder weights are saved.
These are then loaded into SynthesizerTrn.face_enc at the start of TTS finetuning,
giving face_enc a semantically meaningful initialization instead of random noise.

Filelist format (pipe-separated):
  audio_path|image_path|text|phonemes|emotion|level

Usage:
    python pretrain_face_encoder.py \
        --train_filelist /path/to/train.txt \
        --val_filelist   /path/to/val.txt   \
        --output_dir     ./face_enc_pretrain \
        --epochs 50
"""

import os
import argparse
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# import FaceEncoder from the same models.py used by the TTS
from models import FaceEncoder


# ── label maps ────────────────────────────────────────────────────────────────

EMOTIONS = ["angry", "fear", "happy", "surprised", "sad", "neutral"]
LEVELS   = ["level1", "level2", "level3"]

EMOTION2IDX = {e: i for i, e in enumerate(EMOTIONS)}
LEVEL2IDX   = {l: i for i, l in enumerate(LEVELS)}


# ── ordinal loss (same as emotion classifier) ─────────────────────────────────

class OrdinalLoss(nn.Module):
    """
    Encodes intensity as cumulative binary targets:
      level1 → [0, 0]
      level2 → [1, 0]
      level3 → [1, 1]
    """
    NUM_THRESHOLDS = 2

    def __init__(self, level2_weight: float = 2.0):
        super().__init__()
        self.level2_weight = level2_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = labels.size(0)
        targets = torch.zeros(B, 2, device=labels.device)
        targets[labels >= 1, 0] = 1.0
        targets[labels >= 2, 1] = 1.0
        loss = self.bce(logits, targets)
        if self.level2_weight != 1.0:
            w = torch.ones(B, device=labels.device)
            w[labels == 1] = self.level2_weight
            loss = loss * w.unsqueeze(1)
        return loss.mean()

    @staticmethod
    def predict(logits: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(logits) > 0.5).long().sum(dim=1)


# ── classifier model ──────────────────────────────────────────────────────────

class FaceEmotionClassifier(nn.Module):
    """
    FaceEncoder + two classification heads.
    After pretraining, only face_enc weights are transferred to the TTS.
    """

    def __init__(self, num_emotions: int = 6, use_ordinal: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.use_ordinal = use_ordinal

        self.face_enc = FaceEncoder(out_dim=FaceEncoder.OUT_DIM, dropout=dropout)

        hidden = FaceEncoder.OUT_DIM  # 128

        # emotion head
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_emotions),
        )

        # intensity head — emotion-conditioned (mirrors the TTS architecture idea)
        # emotion logits → small projection → concat with face features → intensity
        self.intensity_proj = nn.Sequential(
            nn.Linear(num_emotions, 32),
            nn.ReLU(),
        )
        intensity_out = OrdinalLoss.NUM_THRESHOLDS if use_ordinal else len(LEVELS)
        self.intensity_head = nn.Linear(hidden + 32, intensity_out)

    def forward(self, face: torch.Tensor):
        """
        Args:
            face : (B, 3, 160, 160)  float [0, 255]
        Returns:
            emotion_logits   : (B, num_emotions)
            intensity_logits : (B, 2) if ordinal else (B, 3)
        """
        feat     = self.face_enc(face)                     # (B, 128)
        e_logits = self.emotion_head(feat)                 # (B, 6)

        # intensity conditioned on emotion (detach so intensity loss
        # doesn't affect emotion head gradients)
        e_ctx    = self.intensity_proj(e_logits.detach())  # (B, 32)
        i_input  = torch.cat([feat, e_ctx], dim=-1)        # (B, 160)
        i_logits = self.intensity_head(i_input)            # (B, 2) or (B, 3)

        return e_logits, i_logits


# ── dataset ───────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    """
    Loads face images and emotion/intensity labels from a MEAD filelist.
    Applies aspect-ratio-preserving resize + center crop.
    """

    def __init__(self, filelist_path: str, target_size: int = 160,
                 augment: bool = False):
        self.target_size = target_size
        self.augment     = augment
        self.samples     = self._load(filelist_path)

        # base transform: resize shorter side → center crop → float [0,255]
        self.base_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0),
        ])

        # augmentation: color jitter + horizontal flip
        # NOTE: horizontal flip is safe for faces (emotion doesn't flip)
        self.aug_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0),
        ])

    def _load(self, path: str):
        samples = []
        skipped = 0
        with open(path, "r") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 6:
                    skipped += 1
                    continue
                image_path = parts[1].strip()
                emotion    = parts[4].strip().lower()
                level      = parts[5].strip().lower()

                if emotion not in EMOTION2IDX:
                    skipped += 1; continue
                if level not in LEVEL2IDX:
                    skipped += 1; continue
                if emotion == "neutral":
                    level = "level1"  # neutral only has level1

                if not os.path.isfile(image_path):
                    skipped += 1; continue

                samples.append({
                    "image_path"  : image_path,
                    "emotion_idx" : EMOTION2IDX[emotion],
                    "level_idx"   : LEVEL2IDX[level],
                    "emotion"     : emotion,
                    "level"       : level,
                })
        print(f"[dataset] loaded {len(samples)} samples  (skipped {skipped})")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        transform = self.aug_transform if self.augment else self.base_transform
        face = transform(img)
        return {
            "face"         : face,
            "emotion_label": torch.tensor(s["emotion_idx"], dtype=torch.long),
            "level_label"  : torch.tensor(s["level_idx"],   dtype=torch.long),
        }

    def get_sampler_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler (balance by emotion×level)."""
        keys   = [(s["emotion_idx"], s["level_idx"]) for s in self.samples]
        counts = Counter(keys)
        weights = torch.tensor(
            [1.0 / counts[k] for k in keys], dtype=torch.float)
        return weights


# ── metrics ───────────────────────────────────────────────────────────────────

def unweighted_accuracy(preds, labels, num_classes):
    correct = torch.zeros(num_classes)
    total   = torch.zeros(num_classes)
    for c in range(num_classes):
        mask       = labels == c
        total[c]   = mask.sum().item()
        correct[c] = (preds[mask] == c).sum().item()
    valid = total > 0
    return (correct[valid] / total[valid]).mean().item()


def weighted_accuracy(preds, labels):
    return (preds == labels).float().mean().item()


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, loss_fns, use_ordinal):
    model.eval()
    total_loss = 0.0
    all_e_preds, all_e_labels = [], []
    all_l_preds, all_l_labels = [], []

    for batch in loader:
        face     = batch["face"].to(device)
        e_labels = batch["emotion_label"].to(device)
        l_labels = batch["level_label"].to(device)

        e_logits, i_logits = model(face)

        e_loss = loss_fns["emotion"](e_logits, e_labels)
        l_loss = loss_fns["level"](i_logits, l_labels)
        total_loss += (e_loss + 0.7 * l_loss).item()

        all_e_preds.append(e_logits.argmax(-1).cpu())
        all_e_labels.append(e_labels.cpu())

        if use_ordinal:
            all_l_preds.append(OrdinalLoss.predict(i_logits).cpu())
        else:
            all_l_preds.append(i_logits.argmax(-1).cpu())
        all_l_labels.append(l_labels.cpu())

    e_preds  = torch.cat(all_e_preds)
    e_labels = torch.cat(all_e_labels)
    l_preds  = torch.cat(all_l_preds)
    l_labels = torch.cat(all_l_labels)

    return {
        "val_loss"  : total_loss / len(loader),
        "emotion_UA": unweighted_accuracy(e_preds, e_labels, len(EMOTIONS)),
        "emotion_WA": weighted_accuracy(e_preds, e_labels),
        "level_UA"  : unweighted_accuracy(l_preds, l_labels, len(LEVELS)),
        "level_WA"  : weighted_accuracy(l_preds, l_labels),
    }


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_filelist", required=True)
    p.add_argument("--val_filelist",   required=True)
    p.add_argument("--output_dir",     default="./face_enc_pretrain")
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--use_sampler",    action="store_true", default=True,
                   help="WeightedRandomSampler to balance emotion×level")
    p.add_argument("--level2_weight",  type=float, default=2.0,
                   help="Extra OrdinalLoss weight on level2 samples")
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         default="cuda:0")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pretrain] device: {device}")

    # ── datasets ──────────────────────────────────────────────────────────────
    train_ds = FaceDataset(args.train_filelist, augment=True)
    val_ds   = FaceDataset(args.val_filelist,   augment=False)

    sampler = None
    if args.use_sampler:
        weights = train_ds.get_sampler_weights()
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        print("[pretrain] using WeightedRandomSampler")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # ── model ─────────────────────────────────────────────────────────────────
    model = FaceEmotionClassifier(
        num_emotions=len(EMOTIONS),
        use_ordinal=True,
        dropout=args.dropout,
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[pretrain] params — total: {total/1e6:.2f}M  trainable: {trainable/1e6:.2f}M")

    # ── loss functions ────────────────────────────────────────────────────────
    # class weights for emotion (inverse frequency)
    e_counts = Counter(s["emotion_idx"] for s in train_ds.samples)
    e_w = torch.zeros(len(EMOTIONS))
    for cls, cnt in e_counts.items():
        e_w[cls] = 1.0 / cnt
    e_w = (e_w / e_w.sum() * len(EMOTIONS)).to(device)  # normalised

    loss_fns = {
        "emotion": nn.CrossEntropyLoss(weight=e_w),
        "level"  : OrdinalLoss(level2_weight=args.level2_weight).to(device),
    }

    # ── optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── training loop ─────────────────────────────────────────────────────────
    best_emotion_ua = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            face     = batch["face"].to(device)
            e_labels = batch["emotion_label"].to(device)
            l_labels = batch["level_label"].to(device)

            e_logits, i_logits = model(face)

            e_loss = loss_fns["emotion"](e_logits, e_labels)
            l_loss = loss_fns["level"](i_logits, l_labels)
            loss   = e_loss + 0.7 * l_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        # ── validation ────────────────────────────────────────────────────────
        metrics = evaluate(model, val_loader, device, loss_fns, use_ordinal=True)
        elapsed = time.time() - t0

        print(
            f"[epoch {epoch:3d}/{args.epochs}] "
            f"train_loss={epoch_loss/len(train_loader):.4f}  "
            f"val_loss={metrics['val_loss']:.4f}  "
            f"emotion_UA={metrics['emotion_UA']:.4f}  "
            f"level_UA={metrics['level_UA']:.4f}  "
            f"[{elapsed:.0f}s]"
        )

        # ── save best face_enc weights ────────────────────────────────────────
        if metrics["emotion_UA"] > best_emotion_ua:
            best_emotion_ua = metrics["emotion_UA"]

            # save full classifier checkpoint (for inspection / resuming)
            torch.save({
                "epoch"           : epoch,
                "model_state"     : model.state_dict(),
                "metrics"         : metrics,
                "args"            : vars(args),
            }, out_dir / "best_classifier.pt")

            # save ONLY face_enc weights — this is what goes into the TTS
            torch.save(
                model.face_enc.state_dict(),
                out_dir / "best_face_enc.pt"
            )
            print(f"  ✓ saved best → best_face_enc.pt  "
                  f"(emotion_UA={best_emotion_ua:.4f})")

    print(f"\n[pretrain] done. best emotion_UA = {best_emotion_ua:.4f}")
    print(f"[pretrain] face encoder weights saved to {out_dir / 'best_face_enc.pt'}")
    print(f"[pretrain] use --face_enc_checkpoint {out_dir / 'best_face_enc.pt'} "
          f"when running train_finetune.py")


if __name__ == "__main__":
    main()