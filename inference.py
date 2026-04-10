import os
import torch
import torchaudio
from pathlib import Path

# Thêm đường dẫn hoặc import từ các module đã có
from model import WhisperTransformerPhoneme
from dataset import audio_to_mel
from phoneme_set import INV_VOCAB, CORPUS
import difflib

def main():
    # 1. Cấu hình thiết bị (GPU nếu có, ngược lại CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device}")

    # 2. Khởi tạo mô hình
    # Cần đảm bảo tham số khi khởi tạo model khớp với lúc train.
    # Ở đây mặc định lấy `WhisperTransformerPhoneme`.
    model = WhisperTransformerPhoneme(freeze_encoder=False)
    model.to(device)

    # 3. Nạp trọng số từ checkpoint
    ckpt_path = "checkpoints/whisper_transformer_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"❌ Không tìm thấy checkpoint tại {ckpt_path}!")
        print("💡 Hãy đảm bảo bạn đã train model hoặc cập nhật lại đường dẫn tới file checkpoint (.pt).")
        return

    print(f"Đang nạp checkpoint từ {ckpt_path}...")
    # Nạp checkpoint, dùng map_location để tương thích khi train trên GPU nhưng test trên CPU (hoặc ngược lại)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval() # Đưa model về chế độ suy luận (không cập nhật trọng số)
    
    # 4. Trích xuất đặc trưng và dự đoán từng file audio
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"❌ Thư mục {data_dir} không tồn tại!")
        print("💡 Vui lòng tạo thư mục 'data' và đặt các file audio vào trong đó.")
        return
        
    print("\n" + "="*50)
    print(" KẾT QUẢ DỰ ĐOÁN ")
    print("="*50)
    
    # Xây dựng từ điển để tra ngược từ chuỗi âm vị sang Text
    phoneme_to_text = {}
    for text, (words, phonemes) in CORPUS.items():
        phoneme_str = " ".join(phonemes)
        phoneme_to_text[phoneme_str] = text
    
    # Duyệt file audio từ audio1.wav đến audio10.wav
    for i in range(1, 11):
        audio_name = f"audio{i}.wav"
        audio_path = data_dir / audio_name
        
        if not audio_path.exists():
            print(f"{audio_name:<12} : (File không tồn tại)")
            continue
            
        try:
            # Đọc waveform
            waveform, sr = torchaudio.load(str(audio_path))
            
            # Chẩn đoán lỗi shape nếu file nhiều kênh
            # Hàm audio_to_mel của bạn đã xử lý đưa về mono bên trong nội bộ rồi
            
            # Chuyển đổi thành mel-spectrogram
            mel = audio_to_mel(waveform, sr).to(device) # Shape: (T, N_MELS)
            
            # Hàm dự đoán của model yêu cầu batch shape, nên cần thêm chiều cho batch = 1
            mel = mel.unsqueeze(0) # Shape: (1, T, N_MELS)
            mel_lengths = torch.tensor([mel.shape[1]], dtype=torch.long).to(device)
            
            # Suy luận (dùng greedy_decode có sẵn trong model)
            with torch.no_grad():
                preds = model.greedy_decode(mel, mel_lengths, max_len=100)
                
            # Giải mã ra danh sách ký hiệu âm vị (loại bỏ các token đặc biệt <sot>, <eot>, <pad>)
            # Preds[0] vì batch_size = 1
            predicted_phonemes = []
            for idx in preds[0]:
                token = INV_VOCAB.get(idx, "<unk>")
                predicted_phonemes.append(token)
                
            predicted_phoneme_str = " ".join(predicted_phonemes)
            
            # Khớp với CORPUS để lấy ra Text như ASR
            if predicted_phoneme_str in phoneme_to_text:
                asr_text = phoneme_to_text[predicted_phoneme_str]
            else:
                # Khớp tương đối nếu dự đoán có một số âm sai lệch
                matches = difflib.get_close_matches(predicted_phoneme_str, phoneme_to_text.keys(), n=1, cutoff=0.5)
                if matches:
                    asr_text = phoneme_to_text[matches[0]] + " (gần đúng)"
                else:
                    asr_text = "(Không xác định được)"
            
            print(f"{audio_name:<12} : {asr_text}")
            print(f"             [Phonemes]: {predicted_phoneme_str}")
            
        except Exception as e:
            print(f"{audio_name:<12} : Lỗi xử lý -> {str(e)}")

if __name__ == "__main__":
    main()
