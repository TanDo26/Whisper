# VN-EN Cross-Lingual Phoneme Recognition

Triển khai paper: *"Whisper based Cross-Lingual Phoneme Recognition between Vietnamese and English"*

Hệ thống ASR (Automatic Speech Recognition) đa ngôn ngữ Việt-Anh sử dụng kiến trúc Whisper kết hợp với bộ âm vị đại diện (cross-lingual phoneme set) thống nhất cho cả tiếng Việt và tiếng Anh, hỗ trợ cả 3 dạng dữ liệu: thuần Việt, Vietlish, và IEV (code-switching).

---

## Cấu trúc dự án

```
Demo_Whisper/
├── phoneme_set.py       # Bộ âm vị đại diện (Section 3.1) + CORPUS
├── dataset.py           # Dataset classes: Vietnamese / Vietlish / IEV (Section 4.1)
├── model.py             # 3 kiến trúc: Whisper-Transformer / GRU / LSTM (Section 3.2)
├── train.py             # Vòng lặp huấn luyện + đánh giá PER (Section 4.2 & 4.3)
├── inference.py         # Suy luận từ checkpoint + map phoneme → text (ASR)
├── text2phoneme.py      # Chuyển đổi Text → Phoneme (VN / EN / Vietlish / IEV)
├── download_datasets.py # Tải 4 dataset: VLSP2020, CommonVoice, FOSD, VIVOS
├── requirements.txt     # Danh sách thư viện cần thiết
├── data/                # Thư mục chứa audio test (audio1.wav → audio10.wav)
├── checkpoints/         # Checkpoint model tốt nhất sau mỗi lần train
│   └── whisper_transformer_best.pt
└── dataset/             # Dataset sau khi tải về (tạo bởi download_datasets.py)
    ├── vlsp2020/
    │   ├── audio/
    │   └── manifest.jsonl
    ├── common_voice_vi/
    ├── common_voice_en/
    ├── fosd/
    ├── vivos/
    └── manifest_all.jsonl   ← tổng hợp toàn bộ
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

Các thư viện chính:

| Thư viện | Mục đích |
|----------|----------|
| `torch`, `torchaudio` | Deep learning & xử lý audio |
| `datasets` | Tải VLSP2020, CommonVoice, VIVOS từ HuggingFace |
| `kagglehub` | Tải FOSD từ Kaggle |
| `soundfile` | Đọc/ghi file WAV |
| `tqdm` | Hiển thị tiến trình |
| `sounddevice`, `scipy` | Thu âm trực tiếp |

---

## Workflow

```
[Text corpus]
      │
      ▼
text2phoneme.py   ─── Text → Phoneme (VN / EN / Vietlish / IEV)
      │
      ▼
download_datasets.py ─ Tải VLSP2020 + CommonVoice + FOSD + VIVOS
      │                 Lưu audio/*.wav + manifest.jsonl
      ▼
dataset.py        ─── Đọc manifest, load audio → Mel-Spectrogram
      │
      ▼
train.py          ─── Huấn luyện (AdamW + ExponentialLR, 30 epochs)
      │                Lưu checkpoint tốt nhất → checkpoints/*.pt
      ▼
inference.py      ─── Load checkpoint → Dự đoán file audio (ASR)
                       Phoneme sequence → Text (tra CORPUS)
```

---

## Tải Dataset

### Cài thêm (cho Common Voice):
```bash
huggingface-cli login   # Cần HF token vì Common Voice yêu cầu chấp nhận license
```

### Cài thêm (cho FOSD):
Tải `kaggle.json` từ [kaggle.com/settings](https://www.kaggle.com/settings) đặt tại `~/.kaggle/kaggle.json`

### Chạy tải:
```bash
# Tải tất cả 4 dataset
python download_datasets.py

# Tải từng dataset
python download_datasets.py --sources vlsp2020 vivos
python download_datasets.py --sources fosd
python download_datasets.py --sources common_voice
```

### Cấu trúc manifest.jsonl:
Mỗi dòng là một mẫu dữ liệu dạng JSON:
```json
{"audio": "audio/vlsp2020_train_000001.wav", "text": "xin chào mọi người", "source": "vlsp2020", "split": "train"}
```

---

## Text → Phoneme

```python
from text2phoneme import text_to_phoneme

# Tiếng Việt thuần
ph = text_to_phoneme("xin chào", mode="vi")
# → ['s', 'i', 'n', '-1', '$', 'tʃ', 'a', '-2']

# Tiếng Anh native
ph = text_to_phoneme("hello world", mode="en")
# → ['h', 'ə', 'l', 'oʊ', '-1', '|', 'w', 'ɜː', 'l', 'd', '-1']

# Vietlish (từ Anh đọc kiểu Việt)
ph = text_to_phoneme("inbox", mode="vietlish")
# → ['ɪ', 'n', '$', 'b', 'o', '-4', 'k']

# IEV / Code-switching (tự động phân loại từng từ)
ph = text_to_phoneme("anh đang dùng laptop ở nhà", mode="iev")

# Tự động nhận dạng ngôn ngữ
ph = text_to_phoneme("mình sẽ check email ngay", mode="auto")
```

---

## Huấn luyện (Train)

```bash
# Demo inference với dữ liệu synthetic (không cần dataset thật)
python train.py --mode demo

# So sánh 3 kiến trúc: Whisper-GRU / LSTM / Transformer
python train.py --mode compare --epochs 5 --samples 128

# Huấn luyện với dữ liệu synthetic
python train.py --mode train --epochs 30 --samples 1000

# Huấn luyện trên GPU
python train.py --mode train --epochs 30 --samples 5000 --device cuda
```

### Tùy chọn:

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--mode` | `demo` | `demo` / `train` / `compare` |
| `--epochs` | `5` | Số epoch tối đa |
| `--samples` | `64` | Số mẫu synthetic |
| `--device` | `cpu` / `cuda` | Tự động chọn |

---

## Inference (Dự đoán từ file audio)

Đặt file audio vào thư mục `data/` với tên `audio1.wav` → `audio10.wav`, sau đó:

```bash
python inference.py
```

Output ví dụ:
```
==================================================
 KẾT QUẢ DỰ ĐOÁN
==================================================
audio1.wav   : tôi đang đi học (gần đúng)
               [Phonemes]: t o -1 j $ d a -1 ŋ $ d i -1 $ h ɔ -5 k -1
audio2.wav   : xin chào mọi người
               [Phonemes]: s i n -1 $ tʃ a -2 ...
```

---

## Kiến trúc mô hình (Section 3.2)

```
Audio (WAV, 16kHz)
      │
      ▼
Log Mel-Spectrogram (T × 80)   [frame_shift=20ms, frame_len=25ms]
      │
      ▼
2× Conv1D + GELU
      │
      ▼
Sinusoidal Positional Encoding
      │
      ▼
6× Whisper Encoder Block   ← PhoWhisper pre-trained (có thể FROZEN)
      │  H (T' × d_model)
      ↕ Cross-Attention
      │
      ▼
Token Embedding + Learned Positional Encoding
      │
      ▼
6× Transformer Decoder Block
      │
      ▼
Linear → Softmax → Phoneme sequence
```

### 3 biến thể:

| Model | Decoder | Params | Best PER |
|-------|---------|--------|----------|
| Whisper-GRU | 2-layer GRU | ~37M | - |
| Whisper-LSTM | 2-layer LSTM | ~38M | - |
| **Whisper-Transformer** | 6-layer Transformer | ~46M | **Tốt nhất** |

---

## Kết quả paper (Whisper-Transformer, 46M params)

| Dataset | PER (%) |
|---------|---------|
| FOSD | 16.7 |
| VIVOS | 8.85 |
| CmV (Common Voice) | 13.02 |
| VLSP 2020 | 22.4 |
| IEV (code-switching) | 7.02 |
| Vietlish | 16.21 |
| English native | 28.55 |

---

## Bộ âm vị đại diện (Cross-Lingual Phoneme Set)

### Ký hiệu đặc biệt

| Ký hiệu | Ý nghĩa |
|---------|---------|
| `-1` → `-6` | Thanh điệu tiếng Việt: ngang, huyền, hỏi, sắc, nặng, ngã |
| `$` | Ranh giới âm tiết tiếng Việt |
| `\|` | Ranh giới âm tiết tiếng Anh |
| `—` | Liên kết âm trong tiếng Anh |
| `<sot>` / `<eot>` | Start / End of transcript |
| `<pad>` / `<unk>` | Padding / Unknown token |

### Phoneme tiếng Việt: 53 loại
- **Âm đầu**: b, c, ch, d, đ, g, gh, gi, h, k, kh, l, m, n, ng, ngh, nh, p, ph, qu, r, s, t, th, tr, v, x
- **Âm đệm**: w
- **Nguyên âm**: a, ă, â, e, ê, i, o, ô, ơ, u, ư, y
- **Âm cuối**: c, ch, m, n, ng, nh, p, t
- **Thanh điệu**: -1 (ngang), -2 (huyền), -3 (hỏi), -4 (sắc), -5 (nặng), -6 (ngã)

### Phoneme tiếng Anh bổ sung (không có trong tiếng Việt)
`æ ɒ ʌ ɪ ʊ ə iː uː ɜː ɑː ɔː eɪ aɪ ɔɪ oʊ aʊ ɪə eə ʊə dʒ tʃ ʒ θ ð ŋ f z ʃ`

---

## Cấu hình training (theo paper)

| Tham số | Giá trị |
|---------|---------|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Scheduler | ExponentialLR (γ=0.95) |
| Batch size | 16 |
| Max epochs | 30 |
| Grad clip | 1.0 |
| Label smoothing | 0.1 |
| Sample rate | 16 kHz |
| Frame shift | 20 ms |
| Frame length | 25 ms |
| Mel channels | 80 |
| Metric | PER (Phoneme Error Rate) |
