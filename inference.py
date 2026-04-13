import os
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import WhisperTransformerPhoneme
from dataset import collate_fn
from train import compute_per, decode_labels, get_dataset
from phoneme_set import INV_VOCAB

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on custom test data folder")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/whisper_transformer_best.pt", help="Đường dẫn đến model checkpoint")
    parser.add_argument("--dataset_type", type=str, default="base", help="Loại dataset (mặc định 'base' để đọc custom manifest.jsonl)")
    parser.add_argument("--manifest_path", type=str, default="data/manifest.jsonl", help="Đường dẫn tới file manifest.jsonl")
    parser.add_argument("--audio_root", type=str, default="data/", help="Thư mục chứa các file audio (.wav)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Model architecture parameters (Must match the checkpoint!)
    parser.add_argument("--d_model", type=int, default=512, help="Hidden dimension size")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=8, help="Number of decoder layers")
    parser.add_argument("--ffn_dim", type=int, default=2048, help="Feed-forward network dimension")
    
    args = parser.parse_args()

    device = args.device
    print(f"Sử dụng thiết bị: {device}")

    # 1. Khởi tạo model và load checkpoint
    model = WhisperTransformerPhoneme(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_dec_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        freeze_encoder=False
    )
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Không tìm thấy checkpoint tại {args.checkpoint}!")
        print("Vui lòng ghi rõ tham số --checkpoint nếu nó ở vị trí khác.")
        return

    print(f"Đang nạp checkpoint từ {args.checkpoint}...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    # 2. Xây dựng Dataset & DataLoader
    print(f"Đang nạp dữ liệu từ: {args.manifest_path} (audio tại {args.audio_root})")
    ds = get_dataset(args)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)

    # 3. Chạy inference & Tính PER
    print("\n" + "="*85)
    print(" BẮT ĐẦU TEST (INFERENCE) ")
    print(f" Tổng số file: {len(ds)}")
    print("="*85 + "\n")

    refs_all, hyps_all = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing", leave=False):
            mel          = batch["mel"].to(device)
            mel_lengths  = batch["mel_lengths"].to(device)
            labels       = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            # Dự đoán dãy phoneme
            if hasattr(model, "greedy_decode"):
                preds = model.greedy_decode(mel, mel_lengths, max_len=100)
            else:
                logits = model(mel, mel_lengths, labels, label_lengths)
                preds = [logits[i].argmax(-1).tolist() for i in range(mel.size(0))]

            # Giải mã và tính lỗi cho từng sample trong batch
            for i in range(mel.size(0)):
                ref  = decode_labels(labels[i, :label_lengths[i]])
                hyp  = [INV_VOCAB.get(idx, "?") for idx in preds[i]]
                per  = compute_per([ref], [hyp])
                
                # Trích xuất 'text'
                word = batch["texts"][i] if batch.get("texts") and batch["texts"][i] else "N/A"

                refs_all.append(ref)
                hyps_all.append(hyp)

                print(f"[{len(refs_all):03d}] Text: {word}")
                print(f"      True: {' '.join(ref)}")
                print(f"      Pred: {' '.join(hyp)}")
                print(f"      PER:  {per:.2f}%")
                print("-" * 60)

    final_per = compute_per(refs_all, hyps_all)
    print("\n" + "="*85)
    print(f" HOÀN THÀNH. OVERALL PER: {final_per:.2f}%")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
