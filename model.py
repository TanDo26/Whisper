"""
model.py
Kiến trúc Whisper-Transformer cho nhận diện âm vị Việt–Anh.
Dựa trên Section 3.2 của paper.

Sơ đồ tổng quát:
    Audio (waveform)
        ↓
    Log Mel-Spectrogram (T × 80)
        ↓
    2× CNN + GELU  [Encoder bước 1]
        ↓
    Sinusoidal Positional Encoding
        ↓
    6× Whisper Encoder Block (MHA + FFN)  [Frozen hoặc fine-tune]
        ↓  H = (T, d_model)
    ─────────────────────── Cross-Attention ───────────────────────
        ↓
    Token Embedding + Learned Positional Encoding  [Decoder input]
        ↓
    6× Transformer Decoder Block (Self-Attn + Cross-Attn + FFN)
        ↓
    Linear → Softmax → Âm vị
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from phoneme_set import VOCAB_SIZE, SPECIAL_TOKENS, VOCAB


# ═══════════════════════════════════════════════════════════════════════════
#  1. POSITIONAL ENCODINGS
# ═══════════════════════════════════════════════════════════════════════════

class SinusoidalPositionalEncoding(nn.Module):
    """
    Positional encoding hình sin cho Encoder — dùng trong Whisper gốc.
    Công thức (2) trong paper: X_pos = X_conv + P(X_conv)
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class LearnedPositionalEncoding(nn.Module):
    """Positional encoding học được cho Decoder."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.pe(positions)


# ═══════════════════════════════════════════════════════════════════════════
#  2. ENCODER — PhoWhisper style
# ═══════════════════════════════════════════════════════════════════════════

class WhisperEncoderBlock(nn.Module):
    """
    Một khối encoder Whisper: MHA + FFN với residual.
    Công thức (3)(4): H = MHA(Q,K,V),  H = H + FFN(H)
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.mha  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention với residual
        residual = x
        x = self.ln1(x)
        x, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        x = x + residual

        # FFN với residual
        x = x + self.ffn(self.ln2(x))
        return x


class PhoWhisperEncoder(nn.Module):
    """
    Encoder mô phỏng PhoWhisper (dựa trên Whisper base).

    Luồng xử lý:
        X_mel (T×80) → 2×Conv+GELU → Sinusoidal PE → 6×EncoderBlock → H (T×d_model)

    Trong thực tế: load trọng số từ PhoWhisper pre-trained rồi freeze.
    Ở đây: khởi tạo ngẫu nhiên để demo pipeline.

    Công thức (1): X_conv = GELU(2 × Conv(X_mel))
    Công thức (2): X_pos  = X_conv + P(X_conv)
    """

    def __init__(
        self,
        n_mels:   int = 80,
        d_model:  int = 256,
        n_heads:  int = 4,
        n_layers: int = 6,
        ffn_dim:  int = 1024,
        dropout:  float = 0.1,
        frozen:   bool = True,    # Theo paper: freeze encoder
    ):
        super().__init__()
        self.d_model = d_model
        self.frozen  = frozen

        # 2 lớp CNN (theo Whisper gốc)
        self.conv1 = nn.Conv1d(n_mels,  d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

        self.pos_enc  = SinusoidalPositionalEncoding(d_model)
        self.dropout  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            WhisperEncoderBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.ln_post = nn.LayerNorm(d_model)

        # Freeze nếu dùng pre-trained
        if frozen:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, mel: torch.Tensor,
                mel_lengths: Optional[torch.Tensor] = None) -> tuple:
        """
        Args:
            mel:         (B, T, 80)
            mel_lengths: (B,) — độ dài thực của mỗi sequence

        Returns:
            H:               (B, T', d_model)
            encoder_lengths: (B,) — T' sau stride=2
        """
        # CNN expects (B, C, T)
        x = mel.transpose(1, 2)                 # (B, 80, T)

        # Công thức (1): X_conv = GELU(2 × Conv(X_mel))
        x = F.gelu(self.conv1(x))               # (B, d, T)
        x = F.gelu(self.conv2(x))               # (B, d, T//2)

        x = x.transpose(1, 2)                   # (B, T', d)

        # Công thức (2): X_pos = X_conv + P(X_conv)
        x = self.pos_enc(x)
        x = self.dropout(x)

        # Tính key_padding_mask
        key_padding_mask = None
        encoder_lengths  = None
        if mel_lengths is not None:
            encoder_lengths = (mel_lengths + 1) // 2   # do stride=2
            B, T_enc, _ = x.shape
            mask = torch.arange(T_enc, device=x.device).unsqueeze(0) >= encoder_lengths.unsqueeze(1)
            key_padding_mask = mask

        # Công thức (3)(4): H = MHA + FFN × n_layers
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        H = self.ln_post(x)
        return H, encoder_lengths


# ═══════════════════════════════════════════════════════════════════════════
#  3. DECODER — Transformer với Multiple Cross-Attention
# ═══════════════════════════════════════════════════════════════════════════

class TransformerDecoderBlock(nn.Module):
    """
    Một khối Transformer Decoder:
        Self-Attention (masked) → Cross-Attention → FFN

    Điểm quan trọng theo paper:
        Multiple Cross-Attention xuyên suốt các block để tránh mất thông tin
        encoder khi đi qua nhiều lớp (Eq. 6).

    Công thức (6):
        Cross-Attention(Q_enc, K, V) = softmax(Q_enc·K^T / √dk) · V
        Trong đó Q_enc là encoded audio, K,V từ phoneme sequences.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        # Self-attention (masked — autoregressive)
        self.ln1       = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Cross-attention — Q từ decoder, K/V từ encoder H
        self.ln2        = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # FFN
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:                torch.Tensor,
        encoder_out:      torch.Tensor,
        tgt_mask:         Optional[torch.Tensor] = None,
        tgt_key_pad_mask: Optional[torch.Tensor] = None,
        memory_key_mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Masked Self-Attention
        residual = x
        x = self.ln1(x)
        x, _ = self.self_attn(x, x, x,
                               attn_mask=tgt_mask,
                               key_padding_mask=tgt_key_pad_mask)
        x = self.dropout(x) + residual

        # 2. Cross-Attention (Eq. 6)
        # Q = decoder hidden, K/V = encoder output H
        residual = x
        x = self.ln2(x)
        x, _ = self.cross_attn(x, encoder_out, encoder_out,
                                key_padding_mask=memory_key_mask)
        x = self.dropout(x) + residual

        # 3. FFN
        x = x + self.ffn(self.ln3(x))
        return x


class PhonemeDecoder(nn.Module):
    """
    Transformer Decoder với 6 blocks.
    Autoregressive: mỗi token được dự đoán dựa trên các token trước + encoder output.

    Công thức (5): E = PosEnc(S) + Embedding(S)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int = 256,
        n_heads:    int = 4,
        n_layers:   int = 6,
        ffn_dim:    int = 1024,
        max_len:    int = 512,
        dropout:    float = 0.1,
        pad_idx:    int = 0,
    ):
        super().__init__()
        self.d_model  = d_model
        self.pad_idx  = pad_idx

        # Công thức (5): Embedding + Learned Positional Encoding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = LearnedPositionalEncoding(d_model, max_len)
        self.dropout   = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final  = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Mask tam giác trên để đảm bảo autoregressive."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        tgt:          torch.Tensor,
        encoder_out:  torch.Tensor,
        tgt_lengths:  Optional[torch.Tensor] = None,
        memory_mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt:         (B, L) — chuỗi phoneme target (shifted right)
            encoder_out: (B, T', d_model)
            tgt_lengths: (B,) — độ dài thực của target
            memory_mask: (B, T') — padding mask của encoder

        Returns:
            logits: (B, L, vocab_size)
        """
        B, L = tgt.shape

        # Causal mask
        causal_mask = self._make_causal_mask(L, tgt.device)

        # Padding mask cho target
        tgt_key_pad_mask = None
        if tgt_lengths is not None:
            tgt_key_pad_mask = (
                torch.arange(L, device=tgt.device).unsqueeze(0) >= tgt_lengths.unsqueeze(1)
            )

        # Công thức (5): E = PosEnc(S) + Embedding(S)
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.dropout(x)

        # 6 Decoder blocks — mỗi block có Cross-Attention riêng
        for block in self.blocks:
            x = block(x, encoder_out,
                      tgt_mask=causal_mask,
                      tgt_key_pad_mask=tgt_key_pad_mask,
                      memory_key_mask=memory_mask)

        x = self.ln_final(x)
        logits = self.output_proj(x)    # (B, L, vocab_size)
        return logits


# ═══════════════════════════════════════════════════════════════════════════
#  4. MÔ HÌNH TỔNG HỢP — Whisper-Transformer
# ═══════════════════════════════════════════════════════════════════════════

class WhisperTransformerPhoneme(nn.Module):
    """
    Mô hình Whisper-Transformer cho nhận diện âm vị tiếng Việt–Anh.

    Theo paper (Table 2 & 3):
        - Kích thước: ~46M tham số
        - Encoder: PhoWhisper (frozen)
        - Decoder: 6× Transformer blocks
        - Kết quả tốt nhất trong 4 kiến trúc so sánh:
            FOSD: 16.7%, VIVOS: 8.85%, IEV: 7.02%, En native: 28.55%
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        n_mels:     int = 80,
        d_model:    int = 256,
        n_heads:    int = 4,
        n_enc_layers: int = 6,
        n_dec_layers: int = 6,
        ffn_dim:    int = 1024,
        dropout:    float = 0.1,
        freeze_encoder: bool = True,
        pad_idx:    int = 0,
    ):
        super().__init__()

        self.encoder = PhoWhisperEncoder(
            n_mels=n_mels, d_model=d_model, n_heads=n_heads,
            n_layers=n_enc_layers, ffn_dim=ffn_dim,
            dropout=dropout, frozen=freeze_encoder,
        )

        self.decoder = PhonemeDecoder(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_layers=n_dec_layers, ffn_dim=ffn_dim,
            dropout=dropout, pad_idx=pad_idx,
        )

        self.pad_idx = pad_idx

    def forward(
        self,
        mel:           torch.Tensor,
        mel_lengths:   Optional[torch.Tensor],
        tgt:           torch.Tensor,
        tgt_lengths:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            mel:         (B, T, 80)
            mel_lengths: (B,)
            tgt:         (B, L) — target sequence (with <sot>)
            tgt_lengths: (B,)

        Returns:
            logits: (B, L-1, vocab_size)
        """
        encoder_out, enc_lengths = self.encoder(mel, mel_lengths)

        # Memory key padding mask
        memory_mask = None
        if enc_lengths is not None:
            T_enc = encoder_out.size(1)
            memory_mask = (
                torch.arange(T_enc, device=mel.device).unsqueeze(0)
                >= enc_lengths.unsqueeze(1)
            )

        # Teacher forcing: input = tgt[:-1], target = tgt[1:]
        decoder_input = tgt[:, :-1]
        dec_lengths   = (tgt_lengths - 1) if tgt_lengths is not None else None

        logits = self.decoder(decoder_input, encoder_out, dec_lengths, memory_mask)
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        mel:         torch.Tensor,
        mel_lengths: Optional[torch.Tensor] = None,
        max_len:     int = 100,
    ) -> list:
        """
        Giải mã tham lam (greedy) không dùng beam search.

        Returns:
            list of list of int — chuỗi index âm vị cho mỗi sample
        """
        self.eval()
        encoder_out, enc_lengths = self.encoder(mel, mel_lengths)

        memory_mask = None
        if enc_lengths is not None:
            T_enc = encoder_out.size(1)
            memory_mask = (
                torch.arange(T_enc, device=mel.device).unsqueeze(0)
                >= enc_lengths.unsqueeze(1)
            )

        B     = mel.size(0)
        sot   = VOCAB[SPECIAL_TOKENS["SOT"]]
        eot   = VOCAB[SPECIAL_TOKENS["EOT"]]

        generated = torch.full((B, 1), sot, dtype=torch.long, device=mel.device)
        finished  = torch.zeros(B, dtype=torch.bool, device=mel.device)
        results   = [[] for _ in range(B)]

        for _ in range(max_len):
            logits = self.decoder(generated, encoder_out, memory_mask=memory_mask)
            next_token = logits[:, -1, :].argmax(dim=-1)   # (B,)

            for i in range(B):
                if not finished[i]:
                    tok = next_token[i].item()
                    if tok == eot:
                        finished[i] = True
                    else:
                        results[i].append(tok)

            if finished.all():
                break

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        return results


# ═══════════════════════════════════════════════════════════════════════════
#  5. CÁC MODEL SO SÁNH (theo paper Section 4.2)
# ═══════════════════════════════════════════════════════════════════════════

class WhisperGRUPhoneme(nn.Module):
    """
    Whisper-GRU: PhoWhisper Encoder + MHA + 3 GRU blocks (~30M params).
    Kết quả kém nhất trong paper (VIVOS: 46.45%).
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 256,
                 dropout: float = 0.1, freeze_encoder: bool = True):
        super().__init__()
        self.encoder  = PhoWhisperEncoder(d_model=d_model, frozen=freeze_encoder)
        self.mha      = nn.MultiheadAttention(d_model, 4, dropout=dropout, batch_first=True)
        self.gru      = nn.GRU(d_model, d_model, num_layers=3,
                               batch_first=True, dropout=dropout)
        self.proj     = nn.Linear(d_model, vocab_size)

    def forward(self, mel, mel_lengths=None, tgt=None, tgt_lengths=None):
        H, _ = self.encoder(mel, mel_lengths)
        H, _ = self.mha(H, H, H)
        out, _ = self.gru(H)
        return self.proj(out)


class WhisperLSTMPhoneme(nn.Module):
    """
    Whisper-LSTM: thay GRU bằng LSTM (~32M params).
    Kết quả trung bình (VIVOS: 31.08%).
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = 256,
                 dropout: float = 0.1, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = PhoWhisperEncoder(d_model=d_model, frozen=freeze_encoder)
        self.mha     = nn.MultiheadAttention(d_model, 4, dropout=dropout, batch_first=True)
        self.lstm    = nn.LSTM(d_model, d_model, num_layers=3,
                               batch_first=True, dropout=dropout)
        self.proj    = nn.Linear(d_model, vocab_size)

    def forward(self, mel, mel_lengths=None, tgt=None, tgt_lengths=None):
        H, _ = self.encoder(mel, mel_lengths)
        H, _ = self.mha(H, H, H)
        out, _ = self.lstm(H)
        return self.proj(out)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    B, T, C = 2, 160, 80
    mel     = torch.randn(B, T, C)
    lengths = torch.tensor([160, 120])

    # Target: <sot> + 5 phonemes + <eot>
    tgt     = torch.randint(0, VOCAB_SIZE, (B, 7))
    tgt[:, 0] = VOCAB[SPECIAL_TOKENS["SOT"]]
    tgt[:, -1] = VOCAB[SPECIAL_TOKENS["EOT"]]
    tgt_len = torch.tensor([7, 6])

    print("=== Whisper-Transformer ===")
    model  = WhisperTransformerPhoneme(freeze_encoder=False)
    logits = model(mel, lengths, tgt, tgt_len)
    print(f"Logits shape: {logits.shape}")   # (2, 6, vocab_size)
    print(f"Params: {count_parameters(model):,}")

    print("\n=== Greedy Decode ===")
    preds = model.greedy_decode(mel, lengths, max_len=20)
    for i, p in enumerate(preds):
        from phoneme_set import INV_VOCAB
        tokens = [INV_VOCAB.get(idx, "?") for idx in p]
        print(f"  Sample {i}: {tokens}")

    print("\n=== Model Comparison ===")
    for name, cls in [("Whisper-GRU", WhisperGRUPhoneme),
                      ("Whisper-LSTM", WhisperLSTMPhoneme),
                      ("Whisper-Transformer", WhisperTransformerPhoneme)]:
        m = cls(freeze_encoder=False)
        print(f"  {name:30s}: {count_parameters(m):>10,} trainable params")
