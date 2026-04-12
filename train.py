"""
train.py
Vòng lặp huấn luyện và đánh giá mô hình.
Dựa trên Section 4.2 của paper:
    - Optimizer: AdamW
    - Scheduler: ExponentialLR
    - Learning rate: 0.001
    - Max epochs: 30
    - Batch size: 16
    - Metric: Phoneme Error Rate (PER)
"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from phoneme_set import VOCAB, INV_VOCAB, SPECIAL_TOKENS, VOCAB_SIZE
from dataset import (
    SyntheticVietlishDataset, 
    VietnameseDataset, 
    VietlishDataset, 
    IEVDataset, 
    PhonemeDatasetBase,
    collate_fn
)
from model import WhisperTransformerPhoneme, WhisperGRUPhoneme, WhisperLSTMPhoneme


def get_dataset(args, n_samples=None):
    if args.dataset_type == "synthetic":
        samples = n_samples if n_samples is not None else args.samples
        return SyntheticVietlishDataset(n_samples=samples)
    elif args.dataset_type == "vietnamese":
        return VietnameseDataset(args.manifest_path, args.audio_root)
    elif args.dataset_type == "vietlish":
        return VietlishDataset(args.manifest_path, args.audio_root)
    elif args.dataset_type == "iev":
        return IEVDataset(args.manifest_path, args.audio_root)
    elif args.dataset_type == "base":
        return PhonemeDatasetBase(args.manifest_path, args.audio_root)
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")


# ═══════════════════════════════════════════════════════════════════════════
#  1. PHONEME ERROR RATE (PER)
# ═══════════════════════════════════════════════════════════════════════════

def levenshtein(ref: list, hyp: list) -> int:
    """
    Tính khoảng cách Levenshtein (edit distance) giữa hai chuỗi.
    Được dùng trong công thức PER (Eq. 7).
    """
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_per(references: list, hypotheses: list) -> float:
    """
    Tính Phoneme Error Rate trung bình.

    PER = (I + D + S) / N  (Eq. 7 trong paper)

    Args:
        references:  list of list of tokens (ground truth)
        hypotheses:  list of list of tokens (prediction)

    Returns:
        PER (%) — float
    """
    total_errors = 0
    total_tokens = 0

    for ref, hyp in zip(references, hypotheses):
        total_errors += levenshtein(ref, hyp)
        total_tokens += len(ref)

    if total_tokens == 0:
        return 0.0
    return 100.0 * total_errors / total_tokens


def decode_labels(label_ids: torch.Tensor, pad_idx: int = 0) -> list:
    """Chuyển tensor index → list of token strings, bỏ <sot>/<eot>/<pad>."""
    skip = {
        VOCAB[SPECIAL_TOKENS["SOT"]],
        VOCAB[SPECIAL_TOKENS["EOT"]],
        VOCAB[SPECIAL_TOKENS["PAD"]],
    }
    tokens = []
    for idx in label_ids.tolist():
        if idx in skip:
            continue
        tokens.append(INV_VOCAB.get(idx, "<unk>"))
    return tokens


# ═══════════════════════════════════════════════════════════════════════════
#  2. LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def compute_loss(
    logits:      torch.Tensor,
    tgt:         torch.Tensor,
    tgt_lengths: torch.Tensor,
    pad_idx:     int = 0,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Cross-entropy loss với label smoothing.

    Args:
        logits:      (B, L-1, vocab_size)
        tgt:         (B, L)   — bao gồm <sot> ... <eot>
        tgt_lengths: (B,)
        pad_idx:     index của <pad>

    Returns:
        scalar loss
    """
    # Target: bỏ <sot>, giữ từ index 1 trở đi
    target = tgt[:, 1:]                          # (B, L-1)

    B, L, V = logits.shape
    logits_flat = logits.reshape(-1, V)          # (B*(L-1), V)
    target_flat = target.reshape(-1)             # (B*(L-1),)

    loss = F.cross_entropy(
        logits_flat, target_flat,
        ignore_index=pad_idx,
        label_smoothing=label_smoothing,
    )
    return loss


def _align_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Đảm bảo logits có shape (B, L-1, V) để khớp với target.
    GRU/LSTM trả về (B, T', V) — cần crop/pad về đúng L-1.
    """
    tgt_len = labels.shape[1] - 1
    L = logits.shape[1]
    if L == tgt_len:
        return logits
    if L > tgt_len:
        return logits[:, :tgt_len, :]
    pad = torch.zeros(logits.shape[0], tgt_len - L, logits.shape[2], device=logits.device)
    return torch.cat([logits, pad], dim=1)


# ═══════════════════════════════════════════════════════════════════════════
#  3. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Vòng lặp huấn luyện theo paper:
        - AdamW optimizer, lr=0.001
        - ExponentialLR scheduler
        - Max 30 epochs, batch size 16
        - Audio resampled to 16kHz, frame shift 20ms, frame length 25ms
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        device:       str = "cpu",
        lr:           float = 1e-3,
        gamma:        float = 0.95,      # ExponentialLR decay
        max_epochs:   int   = 30,
        save_dir:     str   = "checkpoints",
        model_name:   str   = "whisper_transformer",
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.max_epochs   = max_epochs
        self.save_dir     = Path(save_dir)
        self.model_name   = model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Chỉ optimize các param không bị freeze
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=gamma
        )

        self.history = {
            "train_loss": [], "val_loss": [], "val_per": []
        }
        self.best_per = float("inf")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.train_loader:
            mel          = batch["mel"].to(self.device)
            mel_lengths  = batch["mel_lengths"].to(self.device)
            labels       = batch["labels"].to(self.device)
            label_lengths = batch["label_lengths"].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(mel, mel_lengths, labels, label_lengths)
            logits = _align_logits(logits, labels)
            loss   = compute_loss(logits, labels, label_lengths)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self) -> tuple:
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0
        all_refs   = []
        all_hyps   = []

        for batch in self.val_loader:
            mel          = batch["mel"].to(self.device)
            mel_lengths  = batch["mel_lengths"].to(self.device)
            labels       = batch["labels"].to(self.device)
            label_lengths = batch["label_lengths"].to(self.device)

            logits = self.model(mel, mel_lengths, labels, label_lengths)
            logits_aligned = _align_logits(logits, labels)
            loss   = compute_loss(logits_aligned, labels, label_lengths)
            total_loss += loss.item()
            n_batches  += 1

            # Greedy decode để tính PER
            if hasattr(self.model, "greedy_decode"):
                preds = self.model.greedy_decode(mel, mel_lengths, max_len=100)
            else:
                preds = [logits[i].argmax(-1).tolist() for i in range(mel.size(0))]

            for i in range(mel.size(0)):
                ref = decode_labels(labels[i, :label_lengths[i]])
                hyp = [INV_VOCAB.get(idx, "<unk>") for idx in preds[i]]
                all_refs.append(ref)
                all_hyps.append(hyp)

        val_loss = total_loss / max(n_batches, 1)
        val_per  = compute_per(all_refs, all_hyps)
        return val_loss, val_per

    def fit(self):
        print(f"\n{'='*60}")
        print(f" Huấn luyện: {self.model_name}")
        print(f" Device: {self.device} | Epochs: {self.max_epochs}")
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f" Trainable params: {trainable:,}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss, val_per = self.evaluate()
            self.scheduler.step()

            elapsed = time.time() - t0
            lr_now  = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"PER={val_per:.2f}% | "
                f"lr={lr_now:.6f} | "
                f"{elapsed:.1f}s"
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_per"].append(val_per)

            # Lưu checkpoint tốt nhất
            if val_per < self.best_per:
                self.best_per = val_per
                self._save_checkpoint(epoch, val_per)

        print(f"\nBest PER: {self.best_per:.2f}%")
        self._save_history()
        return self.history

    def _save_checkpoint(self, epoch: int, per: float):
        path = self.save_dir / f"{self.model_name}_best.pt"
        torch.save({
            "epoch":      epoch,
            "per":        per,
            "model_state": self.model.state_dict(),
            "opt_state":  self.optimizer.state_dict(),
        }, path)
        print(f"  ✓ Checkpoint saved → {path} (PER={per:.2f}%)")

    def _save_history(self):
        path = self.save_dir / f"{self.model_name}_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
#  4. SO SÁNH CÁC MÔ HÌNH (theo Table 2 & 3 trong paper)
# ═══════════════════════════════════════════════════════════════════════════

def compare_models(args):
    """
    Huấn luyện và so sánh 3 kiến trúc.
    Tái hiện Bảng 2 & 3 của paper.
    """
    print("\n" + "="*60)
    print(" So sánh các kiến trúc mô hình (Paper Table 2 & 3)")
    print("="*60)

    ds = get_dataset(args)
    total_samples = len(ds)
    if args.dataset_type == "synthetic":
        n_val = max(16, total_samples // 5)
    else:
        n_val = max(1, int(total_samples * 0.2))
    n_tr  = total_samples - n_val
    train_ds, val_ds = random_split(ds, [n_tr, n_val])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,  batch_size=8, shuffle=False, collate_fn=collate_fn)

    models = {
        "Whisper-GRU":         WhisperGRUPhoneme(freeze_encoder=False),
        "Whisper-LSTM":        WhisperLSTMPhoneme(freeze_encoder=False),
        "Whisper-Transformer": WhisperTransformerPhoneme(freeze_encoder=False),
    }

    results = {}
    for name, model in models.items():
        print(f"\n→ {name}")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
            max_epochs=args.epochs,
            save_dir="checkpoints",
            model_name=name.replace("-", "_").lower(),
        )
        history = trainer.fit()
        results[name] = {
            "best_per":  trainer.best_per,
            "final_per": history["val_per"][-1],
            "params":    sum(p.numel() for p in model.parameters()),
        }

    # In bảng tổng kết
    print("\n" + "="*60)
    print(f" {'Model':<25} {'Params':>10} {'Best PER':>10}")
    print("-"*60)
    for name, r in results.items():
        print(f" {name:<25} {r['params']:>10,} {r['best_per']:>9.2f}%")
    print("="*60)

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  5. INFERENCE / DEMO
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def inference_demo(model: nn.Module, args):
    """Demo inference."""
    device = args.device
    model.eval()
    model.to(device)

    if args.dataset_type == "synthetic":
        ds = get_dataset(args, n_samples=8)
    else:
        ds = get_dataset(args)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    batch  = next(iter(loader))

    mel        = batch["mel"].to(device)
    mel_lengths = batch["mel_lengths"].to(device)
    labels      = batch["labels"].to(device)
    label_lengths = batch["label_lengths"].to(device)

    preds = model.greedy_decode(mel, mel_lengths, max_len=50)

    print("\n=== Inference Demo ===")
    print(f"{'Word':<12} {'Reference':<35} {'Predicted':<35} PER%")
    print("-" * 95)

    refs_all, hyps_all = [], []
    for i in range(mel.size(0)):
        ref  = decode_labels(labels[i, :label_lengths[i]])
        hyp  = [INV_VOCAB.get(idx, "?") for idx in preds[i]]
        per  = compute_per([ref], [hyp])
        word = batch["texts"][i]

        refs_all.append(ref)
        hyps_all.append(hyp)

        ref_str = " ".join(ref)[:33]
        hyp_str = " ".join(hyp)[:33]
        print(f"{word:<12} {ref_str:<35} {hyp_str:<35} {per:.1f}%")

    print(f"\nOverall PER: {compute_per(refs_all, hyps_all):.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VN-EN Phoneme Recognition Training")
    parser.add_argument("--mode",    choices=["train", "compare", "demo"], default="demo")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs",  type=int, default=5)
    parser.add_argument("--samples", type=int, default=64)
    
    parser.add_argument("--dataset_type", choices=["synthetic", "vietnamese", "vietlish", "iev", "base"], default="synthetic", help="Loại dataset: synthetic, vietnamese, vietlish, iev, base")
    parser.add_argument("--manifest_path", type=str, default="dataset/manifest.jsonl", help="Đường dẫn tới file manifest")
    parser.add_argument("--audio_root", type=str, default="dataset/", help="Đường dẫn tới thư   mục chứa audio")
    
    args = parser.parse_args()

    print(f"Device: {args.device}")

    if args.mode == "demo":
        model = WhisperTransformerPhoneme(freeze_encoder=False)
        inference_demo(model, args)

    elif args.mode == "train":
        ds = get_dataset(args)
        total_samples = len(ds)
        if args.dataset_type == "synthetic":
            n_val = max(8, total_samples // 5)
        else:
            n_val = max(1, int(total_samples * 0.2))
        n_tr = total_samples - n_val
        tr_ds, val_ds = random_split(ds, [n_tr, n_val])

        trainer = Trainer(
            model=WhisperTransformerPhoneme(freeze_encoder=False),
            train_loader=DataLoader(tr_ds,  batch_size=8, shuffle=True,  collate_fn=collate_fn),
            val_loader  =DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn),
            device=args.device,
            max_epochs=args.epochs,
        )
        trainer.fit()

    elif args.mode == "compare":
        compare_models(args)
