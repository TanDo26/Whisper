# -*- coding: utf-8 -*-
"""
download_datasets.py
Tải toàn bộ 4 dataset (VLSP2020, CommonVoice, FOSD, VIVOS)
và lưu vào thư mục `dataset/` theo định dạng thống nhất:
    dataset/
        vlsp2020/    audio/*.wav  +  manifest.jsonl
        common_voice_vi/  audio/*.wav  +  manifest.jsonl
        common_voice_en/  audio/*.wav  +  manifest.jsonl
        fosd/        audio/*.wav  +  manifest.jsonl
        vivos/       audio/*.wav  +  manifest.jsonl

Mỗi dòng trong manifest.jsonl có dạng:
    {"audio": "audio/utt0001.wav", "text": "xin chào", "source": "vlsp2020"}
"""

import os
import sys
import json
import shutil
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Thư mục lưu toàn bộ
DATASET_ROOT = Path("dataset")
DATASET_ROOT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: Lưu audio (numpy array) ra file .wav
# ─────────────────────────────────────────────────────────────────────────────

def save_audio(audio_array, sample_rate: int, out_path: Path):
    """Lưu numpy array hoặc bytes thành file WAV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(audio_array, bytes):
        # Trường hợp HuggingFace trả về bytes thô
        with open(out_path, "wb") as f:
            f.write(audio_array)
    else:
        if hasattr(audio_array, "numpy"):
            audio_array = audio_array.numpy()
        audio_array = np.array(audio_array, dtype=np.float32)
        sf.write(str(out_path), audio_array, sample_rate)


def write_manifest(records: list, manifest_path: Path):
    """Ghi danh sách records ra file manifest.jsonl."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  >> Saved manifest: {manifest_path} ({len(records)} records)")


# ─────────────────────────────────────────────────────────────────────────────
#  1. VLSP 2020
# ─────────────────────────────────────────────────────────────────────────────

def download_vlsp2020():
    print("\n" + "="*60)
    print("  [1/4] Downloading VLSP 2020 ...")
    print("="*60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] Missing 'datasets'. Run: pip install datasets")
        return

    out_dir   = DATASET_ROOT / "vlsp2020"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("doof-ferb/vlsp2020_vinai_100h", trust_remote_code=True)

    records = []
    idx = 0

    for split_name, split_ds in ds.items():
        print(f"  Processing split: {split_name} ({len(split_ds)} samples)")
        for item in tqdm(split_ds, desc=f"    VLSP2020/{split_name}"):
            fname  = f"vlsp2020_{split_name}_{idx:06d}.wav"
            fpath  = audio_dir / fname

            audio  = item["audio"]
            save_audio(audio["array"], audio["sampling_rate"], fpath)

            records.append({
                "audio":  f"audio/{fname}",
                "text":   item.get("sentence", item.get("text", "")),
                "source": "vlsp2020",
                "split":  split_name,
            })
            idx += 1

    write_manifest(records, out_dir / "manifest.jsonl")
    print(f"  [VLSP2020] Done: {idx} samples saved to {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  2. COMMON VOICE (Vietnamese + English)
# ─────────────────────────────────────────────────────────────────────────────

def download_common_voice():
    print("\n" + "="*60)
    print("  [2/4] Downloading Common Voice (vi + en) ...")
    print("="*60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] Missing 'datasets'. Run: pip install datasets")
        return

    for lang_code, lang_name in [("vi", "common_voice_vi"), ("en", "common_voice_en")]:
        print(f"\n  -- Language: {lang_code} --")

        out_dir   = DATASET_ROOT / lang_name
        audio_dir = out_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Common Voice 17 là phiên bản mới nhất có sẵn
            ds = load_dataset(
                "mozilla-foundation/common_voice_17_0",
                lang_code,
                trust_remote_code=True,
                token=True,   # Cần HuggingFace token cho Common Voice (CC license)
            )
        except Exception as e:
            print(f"  [WARN] Could not load common_voice_17_0/{lang_code}: {e}")
            print(f"  Trying common_voice_11_0 ...")
            try:
                ds = load_dataset(
                    "mozilla-foundation/common_voice_11_0",
                    lang_code,
                    trust_remote_code=True,
                )
            except Exception as e2:
                print(f"  [ERROR] Failed to load Common Voice {lang_code}: {e2}")
                continue

        records = []
        idx = 0

        for split_name, split_ds in ds.items():
            print(f"    Processing split: {split_name} ({len(split_ds)} samples)")
            for item in tqdm(split_ds, desc=f"    CmV-{lang_code}/{split_name}"):
                fname  = f"cv_{lang_code}_{split_name}_{idx:06d}.wav"
                fpath  = audio_dir / fname

                audio  = item["audio"]
                save_audio(audio["array"], audio["sampling_rate"], fpath)

                records.append({
                    "audio":    f"audio/{fname}",
                    "text":     item.get("sentence", ""),
                    "source":   f"common_voice_{lang_code}",
                    "split":    split_name,
                    "language": lang_code,
                })
                idx += 1

        write_manifest(records, out_dir / "manifest.jsonl")
        print(f"  [CommonVoice-{lang_code}] Done: {idx} samples saved to {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  3. FOSD (FPT Open Speech Dataset)  via Kaggle
# ─────────────────────────────────────────────────────────────────────────────

def download_fosd():
    print("\n" + "="*60)
    print("  [3/4] Downloading FOSD via KaggleHub ...")
    print("="*60)

    try:
        import kagglehub
    except ImportError:
        print("  [ERROR] Missing 'kagglehub'. Run: pip install kagglehub")
        return

    try:
        raw_path = kagglehub.dataset_download(
            "thinh127/fpt-open-speech-dataset-fosd-vietnamese"
        )
    except Exception as e:
        print(f"  [ERROR] KaggleHub download failed: {e}")
        print("  Make sure kaggle.json is configured (~/.kaggle/kaggle.json)")
        return

    raw_path = Path(raw_path)
    print(f"  Raw FOSD files downloaded to: {raw_path}")

    out_dir   = DATASET_ROOT / "fosd"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    records = []
    idx = 0

    # Tìm tất cả file audio (wav/mp3/flac) trong thư mục tải về
    audio_exts = {".wav", ".mp3", ".flac", ".ogg"}
    audio_files = [f for f in raw_path.rglob("*") if f.suffix.lower() in audio_exts]

    # Tìm transcript nếu có (thường là file .txt hoặc .json cùng thư mục)
    # FOSD thường đặt transcript trong cùng thư mục với tên file giống nhau
    print(f"  Found {len(audio_files)} audio files in FOSD")

    for src_audio in tqdm(audio_files, desc="    FOSD"):
        fname      = f"fosd_{idx:06d}.wav"
        dst_audio  = audio_dir / fname

        # Copy/convert sang wav
        if src_audio.suffix.lower() == ".wav":
            shutil.copy2(src_audio, dst_audio)
        else:
            try:
                data, sr = sf.read(str(src_audio))
                sf.write(str(dst_audio), data, sr)
            except Exception as e:
                print(f"  [WARN] Could not convert {src_audio.name}: {e}")
                idx += 1
                continue

        # Tìm transcript tương ứng
        txt_candidates = [
            src_audio.with_suffix(".txt"),
            src_audio.with_suffix(".lab"),
        ]
        transcript = ""
        for tc in txt_candidates:
            if tc.exists():
                transcript = tc.read_text(encoding="utf-8").strip()
                break

        records.append({
            "audio":  f"audio/{fname}",
            "text":   transcript,
            "source": "fosd",
            "original_file": src_audio.name,
        })
        idx += 1

    write_manifest(records, out_dir / "manifest.jsonl")
    print(f"  [FOSD] Done: {idx} samples saved to {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  4. VIVOS
# ─────────────────────────────────────────────────────────────────────────────

def download_vivos():
    print("\n" + "="*60)
    print("  [4/4] Downloading VIVOS ...")
    print("="*60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] Missing 'datasets'. Run: pip install datasets")
        return

    out_dir   = DATASET_ROOT / "vivos"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset("ademax/vivos-vie-speech2text", trust_remote_code=True)
    except Exception as e:
        print(f"  [ERROR] Failed to load VIVOS: {e}")
        print("  Trying official vivos dataset...")
        try:
            ds = load_dataset("vivos", trust_remote_code=True)
        except Exception as e2:
            print(f"  [ERROR] Failed: {e2}")
            return

    records = []
    idx = 0

    for split_name, split_ds in ds.items():
        print(f"  Processing split: {split_name} ({len(split_ds)} samples)")
        for item in tqdm(split_ds, desc=f"    VIVOS/{split_name}"):
            fname  = f"vivos_{split_name}_{idx:06d}.wav"
            fpath  = audio_dir / fname

            audio  = item["audio"]
            save_audio(audio["array"], audio["sampling_rate"], fpath)

            # VIVOS có thể dùng "sentence", "transcription" hoặc "text"
            text = (
                item.get("sentence")
                or item.get("transcription")
                or item.get("text")
                or ""
            )

            records.append({
                "audio":  f"audio/{fname}",
                "text":   text,
                "source": "vivos",
                "split":  split_name,
            })
            idx += 1

    write_manifest(records, out_dir / "manifest.jsonl")
    print(f"  [VIVOS] Done: {idx} samples saved to {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  TẠO FILE MANIFEST TỔNG HỢP (merged)
# ─────────────────────────────────────────────────────────────────────────────

def merge_all_manifests():
    """Gộp tất cả manifest.jsonl từ các sub-dataset thành 1 file duy nhất."""
    print("\n" + "="*60)
    print("  [MERGE] Combining all manifests ...")
    print("="*60)

    merged_path = DATASET_ROOT / "manifest_all.jsonl"
    all_records = []

    for sub_dir in sorted(DATASET_ROOT.iterdir()):
        manifest = sub_dir / "manifest.jsonl"
        if not manifest.exists():
            continue

        sub_records = []
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Thêm đường dẫn tương đối từ gốc dataset/
                rec["audio"] = f"{sub_dir.name}/{rec['audio']}"
                sub_records.append(rec)

        print(f"  {sub_dir.name:25s}: {len(sub_records):6d} records")
        all_records.extend(sub_records)

    with open(merged_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n  >> Merged manifest: {merged_path}")
    print(f"     Total records   : {len(all_records)}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download all 4 ASR datasets")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["vlsp2020", "common_voice", "fosd", "vivos"],
        choices=["vlsp2020", "common_voice", "fosd", "vivos"],
        help="Chọn dataset muốn tải (mặc định: tải tất cả)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  DATASET DOWNLOADER")
    print(f"  Save location: {DATASET_ROOT.resolve()}")
    print("=" * 60)

    if "vlsp2020"     in args.sources: download_vlsp2020()
    if "common_voice" in args.sources: download_common_voice()
    if "fosd"         in args.sources: download_fosd()
    if "vivos"        in args.sources: download_vivos()

    merge_all_manifests()

    print("\n" + "=" * 60)
    print("  DONE! Dataset structure:")
    print(f"  {DATASET_ROOT}/")
    for d in sorted(DATASET_ROOT.iterdir()):
        if d.is_dir():
            n = sum(1 for _ in (d / "audio").glob("*.wav")) if (d / "audio").exists() else 0
            print(f"    {d.name}/  ({n} wav files)")
    print(f"    manifest_all.jsonl  (merged)")
    print("=" * 60)
