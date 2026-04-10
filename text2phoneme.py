"""
text2phoneme.py
Chuyển đổi văn bản (Text) → Chuỗi âm vị (Phoneme Sequence)
theo phương pháp trong paper:
    - Tiếng Việt: rule-based syllable decomposition (Initial + Rhyme + Tone)
    - Tiếng Anh (Native): G2P mapping → map về cross-lingual phoneme set
    - Vietlish: convert EN word → cách đọc tiếng Việt → phoneme tiếng Việt
    - IEV (code-switching): tokenize câu → xử lý từng từ theo ngôn ngữ đúng
"""

import re
import unicodedata

# ═══════════════════════════════════════════════════════════════════════════
#  1. PHẦN TIẾNG VIỆT: RULE-BASED SYLLABLE DECOMPOSITION
#  Mỗi âm tiết → (Âm đầu, Âm đệm, Nguyên âm, Âm cuối, Thanh điệu)
# ═══════════════════════════════════════════════════════════════════════════

# Bảng tra thanh điệu theo ký tự dấu Unicode
# Unicodedata.name sẽ trả về tên chứa COMBINING... khi nhân biết tonemarks
TONE_MAP = {
    "COMBINING GRAVE ACCENT":             "-2",  # huyền  `
    "COMBINING HOOK ABOVE":               "-3",  # hỏi    ̉
    "COMBINING ACUTE ACCENT":             "-4",  # sắc    ́
    "COMBINING DOT BELOW":                "-5",  # nặng   ̣
    "COMBINING TILDE":                    "-6",  # ngã    ̃
    # ngang (không dấu) → "-1" (default)
}

# Bảng chuyển âm đầu chữ → phoneme
VN_INITIAL_MAP = {
    "b":   "b",  "c":   "k",  "ch":  "tʃ", "d":   "z",  "đ":   "d",
    "g":   "g",  "gh":  "g",  "gi":  "z",  "h":   "h",  "k":   "k",
    "kh":  "x",  "l":   "l",  "m":   "m",  "n":   "n",  "ng":  "ŋ",
    "ngh": "ŋ",  "nh":  "ɲ",  "p":   "p",  "ph":  "f",  "qu":  "k",
    "r":   "r",  "s":   "s",  "t":   "t",  "th":  "tʰ", "tr":  "tʃ",
    "v":   "v",  "x":   "s",
}

# Bảng chuyển vần (rhyme) chữ → chuỗi phoneme
# Bao gồm cả âm đệm + nguyên âm + âm cuối
VN_RHYME_MAP = {
    # ── Nguyên âm đơn ──────────────────────────────────────────
    "a":   ["a"],      "ă":   ["a"],      "â":   ["ɤ"],
    "e":   ["ɛ"],      "ê":   ["e"],      "i":   ["i"],
    "o":   ["ɔ"],      "ô":   ["o"],      "ơ":   ["ɤ"],
    "u":   ["u"],      "ư":   ["ɯ"],      "y":   ["i"],
    # ── Nguyên âm đôi ──────────────────────────────────────────
    "ia":  ["i","a"],  "iê":  ["i","e"],  "ie":  ["i","e"],
    "ua":  ["u","a"],  "uô":  ["u","o"],  "uo":  ["u","o"],
    "ưa":  ["ɯ","a"],  "ươ":  ["ɯ","o"],  "uơ":  ["ɯ","o"],
    "oa":  ["w","a"],  "oă":  ["w","a"],  "oe":  ["w","ɛ"],
    "oo":  ["ɔ"],      "ô":   ["o"],
    "uy":  ["w","i"],  "ui":  ["u","i"],  "oi":  ["ɔ","j"],
    "ôi":  ["o","j"],  "ơi":  ["ɤ","j"],  "ai":  ["a","j"],
    "ay":  ["a","j"],  "ây":  ["ɤ","j"],  "au":  ["a","w"],
    "âu":  ["ɤ","w"],  "ao":  ["a","w"],  "eo":  ["ɛ","w"],
    "êu":  ["e","w"],  "iu":  ["i","w"],  "ưu":  ["ɯ","w"],
    # ── Vần khép (có âm cuối) ──────────────────────────────────
    "an":  ["a","n"],  "ăn":  ["a","n"],  "ân":  ["ɤ","n"],
    "en":  ["ɛ","n"],  "ên":  ["e","n"],  "in":  ["i","n"],
    "on":  ["ɔ","n"],  "ôn":  ["o","n"],  "ơn":  ["ɤ","n"],
    "un":  ["u","n"],  "ưn":  ["ɯ","n"],  "yn":  ["i","n"],
    "am":  ["a","m"],  "ăm":  ["a","m"],  "âm":  ["ɤ","m"],
    "em":  ["ɛ","m"],  "êm":  ["e","m"],  "im":  ["i","m"],
    "om":  ["ɔ","m"],  "ôm":  ["o","m"],  "ơm":  ["ɤ","m"],
    "um":  ["u","m"],  "ưm":  ["ɯ","m"],
    "ang":  ["a","ŋ"],  "ăng": ["a","ŋ"],  "âng": ["ɤ","ŋ"],
    "eng":  ["ɛ","ŋ"],  "êng": ["e","ŋ"],  "ing":  ["i","ŋ"],
    "ong":  ["ɔ","ŋ"],  "ông": ["o","ŋ"],  "ơng": ["ɤ","ŋ"],
    "ung":  ["u","ŋ"],  "ưng": ["ɯ","ŋ"],
    "anh": ["a","ŋ"],  "ênh": ["e","ŋ"],  "inh": ["i","ŋ"],
    "ach": ["a","k"],  "êch": ["e","k"],  "ich": ["i","k"],
    "ao":  ["a","w"],  "eo":  ["ɛ","w"],
    "at":  ["a","t"],  "ăt":  ["a","t"],  "ât":  ["ɤ","t"],
    "et":  ["ɛ","t"],  "êt":  ["e","t"],  "it":  ["i","t"],
    "ot":  ["ɔ","t"],  "ôt":  ["o","t"],  "ơt":  ["ɤ","t"],
    "ut":  ["u","t"],  "ưt":  ["ɯ","t"],
    "ac":  ["a","k"],  "ăc":  ["a","k"],  "âc":  ["ɤ","k"],
    "ec":  ["ɛ","k"],  "oc":  ["ɔ","k"],  "ôc":  ["o","k"],
    "uc":  ["u","k"],  "ưc":  ["ɯ","k"],
    "ap":  ["a","p"],  "ăp":  ["a","p"],  "âp":  ["ɤ","p"],
    "ep":  ["ɛ","p"],  "êp":  ["e","p"],  "ip":  ["i","p"],
    "op":  ["ɔ","p"],  "ôp":  ["o","p"],  "ưp":  ["ɯ","p"],
    "up":  ["u","p"],
    # ── Có âm đệm w ────────────────────────────────────────────
    "oan":  ["w","a","n"],  "oăn": ["w","a","n"],  "oen":  ["w","ɛ","n"],
    "oan":  ["w","a","n"],  "uân":  ["w","ɤ","n"],  "uyên": ["w","i","e","n"],
    "uynh": ["w","i","ŋ"],  "oang": ["w","a","ŋ"],  "uang":  ["w","a","ŋ"],
    "uân":  ["w","ɤ","n"],  "uất":  ["w","ɤ","t"],  "uất":  ["w","ɤ","t"],
    "oat":  ["w","a","t"],  "oac":  ["w","a","k"],  "uat":  ["w","a","t"],
    "uyn":  ["w","i","n"],  "uynh": ["w","i","ŋ"],
    "ươn": ["ɯ","o","n"],   "ướng": ["ɯ","o","ŋ"],  "ường": ["ɯ","o","ŋ"],
    "ươt": ["ɯ","o","t"],   "ướp":  ["ɯ","o","p"],  "ươi": ["ɯ","o","j"],
    "uôn": ["u","o","n"],   "uông": ["u","o","ŋ"],  "uôt": ["u","o","t"],
    "iên": ["i","e","n"],   "iêng": ["i","e","ŋ"],  "iêt": ["i","e","t"],
    "iêm": ["i","e","m"],   "iêp":  ["i","e","p"],  "iêc": ["i","e","k"],
}

# Bảng chuyển thanh điệu theo mark trên nguyên âm
# Dùng cho các nguyên âm có mark (NFD decompose)
TONE_MARK_PATTERN = re.compile(r"[\u0300\u0301\u0303\u0309\u0323]")

def get_tone(syllable_nfd: str) -> str:
    """Trích xuất thanh điệu từ âm tiết đã NFD-decompose."""
    for ch in syllable_nfd:
        name = unicodedata.name(ch, "")
        if name in TONE_MAP:
            return TONE_MAP[name]
    return "-1"  # ngang (không dấu)

def remove_tone_marks(text: str) -> str:
    """Bỏ tất cả dấu thanh, trả về chữ thuần không dấu (nhưng còn dấu phụ âm)."""
    nfd = unicodedata.normalize("NFD", text)
    result = "".join(ch for ch in nfd if not TONE_MARK_PATTERN.match(ch))
    return unicodedata.normalize("NFC", result)

def normalize_vn_base(text: str) -> str:
    """Chuẩn hóa chữ tiếng Việt không dấu thanh, giữ dấu phụ âm."""
    return remove_tone_marks(text.lower())


def vn_syllable_to_phoneme(syllable: str) -> list:
    """
    Phân tích 1 âm tiết tiếng Việt → danh sách phoneme.
    Kết quả: [âm_đầu?, âm_đệm?, nguyên_âm, âm_cuối?, thanh_điệu]

    Ví dụ:
        "chào"  → ["tʃ", "a", "-2"]
        "thống" → ["tʰ", "o", "ŋ", "-4"]
        "anh"   → ["a", "ŋ", "-1"]
    """
    syllable = syllable.strip().lower()
    if not syllable:
        return []

    # Lấy thanh điệu trước khi strip dấu
    nfd = unicodedata.normalize("NFD", syllable)
    tone = get_tone(nfd)

    # Bỏ dấu thanh, giữ dấu phụ âm (â, ê, ô, ă, ơ, ư, đ)
    base = remove_tone_marks(syllable)

    phonemes = []

    # --- Tách âm đầu ---
    initial_ph = None
    remainder = base

    # Thử các âm đầu dài trước (tránh nhầm "gh" thành "g")
    for init in sorted(VN_INITIAL_MAP.keys(), key=len, reverse=True):
        if base.startswith(init):
            initial_ph = VN_INITIAL_MAP[init]
            remainder = base[len(init):]
            # Trường hợp đặc biệt "qu" → âm đệm /w/ kèm theo
            if init == "qu":
                phonemes += [initial_ph, "w"]
                remainder = remainder.lstrip("u")  # bỏ chữ u sau qu
                initial_ph = None  # đã thêm rồi
            break

    if initial_ph:
        phonemes.append(initial_ph)

    # --- Tách rhyme (vần) ---
    # Tìm vần dài nhất khớp với remainder
    rhyme_ph = None
    for rhyme in sorted(VN_RHYME_MAP.keys(), key=len, reverse=True):
        if remainder == rhyme or remainder.startswith(rhyme):
            rhyme_ph = VN_RHYME_MAP[rhyme]
            break

    if rhyme_ph:
        phonemes += rhyme_ph
    else:
        # Fallback: dùng từng ký tự
        for ch in remainder:
            if ch in VN_RHYME_MAP:
                phonemes += VN_RHYME_MAP[ch]
            elif ch == "đ":
                phonemes.append("d")
            else:
                phonemes.append(ch)

    # --- Thêm thanh điệu ---
    phonemes.append(tone)

    return phonemes


def vn_text_to_phoneme(text: str) -> list:
    """
    Chuyển câu tiếng Việt → danh sách phoneme.
    Các âm tiết được phân tách bằng dấu '$' (VN_SEP theo paper).

    Input:  "xin chào"
    Output: ["s", "i", "n", "-1", "$", "tʃ", "a", "-2"]
    """
    words = text.strip().lower().split()
    all_phonemes = []
    for i, word in enumerate(words):
        # Bỏ dấu câu ở đầu/cuối
        word_clean = re.sub(r"^[^\w\u00C0-\u1EF9]+|[^\w\u00C0-\u1EF9]+$", "", word)
        if not word_clean:
            continue
        ph = vn_syllable_to_phoneme(word_clean)
        all_phonemes += ph
        if i < len(words) - 1:
            all_phonemes.append("$")  # ranh giới âm tiết tiếng Việt

    return all_phonemes


# ═══════════════════════════════════════════════════════════════════════════
#  2. PHẦN TIẾNG ANH (NATIVE): G2P MAPPING → CROSS-LINGUAL PHONEME SET
#  Dùng CMUdict-style lookup + fallback rule-based
# ═══════════════════════════════════════════════════════════════════════════

# G2P dictionary nhỏ (Cambridge-style) map sang cross-lingual phoneme set của paper
# Mở rộng dần nếu cần
EN_G2P_DICT = {
    # ── Common words ────────────────────────────────────────────────────────
    "a":       ["ə", "-1"],
    "the":     ["ð", "ə", "-1"],
    "is":      ["ɪ", "z", "-1"],
    "this":    ["ð", "ɪ", "s", "-1"],
    "hello":   ["h", "ə", "l", "oʊ", "-1"],
    "world":   ["w", "ɜː", "l", "d", "-1"],
    "phone":   ["f", "oʊ", "n", "-1"],
    "email":   ["iː", "m", "eɪ", "l", "-1"],
    "message": ["m", "ɛ", "s", "ɪ", "dʒ", "-1"],
    "inbox":   ["ɪ", "n", "b", "ɒ", "k", "s", "-1"],
    "coffee":  ["k", "ɒ", "f", "iː", "-1"],
    "laptop":  ["l", "æ", "p", "t", "ɒ", "p", "-1"],
    "online":  ["ɒ", "n", "l", "aɪ", "n", "-1"],
    "meeting": ["m", "iː", "t", "ɪ", "ŋ", "-1"],
    "video":   ["v", "ɪ", "d", "ɪ", "oʊ", "-1"],
    "check":   ["tʃ", "ɛ", "k", "-1"],
    "share":   ["ʃ", "eə", "-1"],
    "like":    ["l", "aɪ", "k", "-1"],
    "download":["d", "aʊ", "n", "l", "oʊ", "d", "-1"],
    "upload":  ["ʌ", "p", "l", "oʊ", "d", "-1"],
    "wifi":    ["w", "aɪ", "f", "aɪ", "-1"],
    "network": ["n", "ɛ", "t", "w", "ɜː", "k", "-1"],
    "data":    ["d", "eɪ", "t", "ə", "-1"],
    "software":["s", "ɒ", "f", "t", "w", "eə", "-1"],
    "computer":["k", "ə", "m", "p", "juː", "t", "ə", "-1"],
    "internet":["ɪ", "n", "t", "ə", "n", "ɛ", "t", "-1"],
    "system":  ["s", "ɪ", "s", "t", "ə", "m", "-1"],
    "password":["p", "ɑː", "s", "w", "ɜː", "d", "-1"],
    "account": ["ə", "k", "aʊ", "n", "t", "-1"],
    "app":     ["æ", "p", "-1"],
    "chat":    ["tʃ", "æ", "t", "-1"],
    "google":  ["g", "uː", "g", "ə", "l", "-1"],
    "facebook":["f", "eɪ", "s", "b", "ʊ", "k", "-1"],
    "twitter": ["t", "w", "ɪ", "t", "ə", "-1"],
    "youtube": ["j", "uː", "t", "juː", "b", "-1"],
    "school":  ["s", "k", "uː", "l", "-1"],
    "home":    ["h", "oʊ", "m", "-1"],
    "work":    ["w", "ɜː", "k", "-1"],
    "food":    ["f", "uː", "d", "-1"],
    "water":   ["w", "ɔː", "t", "ə", "-1"],
    "money":   ["m", "ʌ", "n", "ɪ", "-1"],
    "time":    ["t", "aɪ", "m", "-1"],
    "day":     ["d", "eɪ", "-1"],
    "year":    ["j", "ɪə", "-1"],
}

# Quy tắc G2P cơ bản để fallback khi không có trong dict
EN_LETTER_PHONEME = {
    "a": "æ", "b": "b", "c": "k", "d": "d", "e": "ɛ",
    "f": "f", "g": "g", "h": "h", "i": "ɪ", "j": "dʒ",
    "k": "k", "l": "l", "m": "m", "n": "n", "o": "ɒ",
    "p": "p", "q": "k", "r": "r", "s": "s", "t": "t",
    "u": "ʌ", "v": "v", "w": "w", "x": "k", "y": "j",
    "z": "z",
}

def en_word_to_phoneme(word: str) -> list:
    """
    Chuyển từ tiếng Anh (native pronunciation) → danh sách phoneme.
    Ưu tiên tra G2P dict, fallback sang letter-by-letter.
    """
    word = word.lower().strip()
    if word in EN_G2P_DICT:
        return EN_G2P_DICT[word]

    # Fallback: map từng chữ cái
    phonemes = []
    i = 0
    while i < len(word):
        # Thử digraph trước
        if i + 1 < len(word):
            digraph = word[i:i+2]
            digraph_map = {
                "sh": "ʃ", "ch": "tʃ", "th": "θ", "ph": "f",
                "wh": "w", "gh": "g", "ng": "ŋ", "ck": "k",
                "qu": "kw",
            }
            if digraph in digraph_map:
                phonemes.append(digraph_map[digraph])
                i += 2
                continue
        ch = word[i]
        if ch in EN_LETTER_PHONEME:
            phonemes.append(EN_LETTER_PHONEME[ch])
        i += 1

    phonemes.append("-1")  # tiếng Anh native dùng tone -1 (ngang/neutral)
    return phonemes


def en_text_to_phoneme(text: str) -> list:
    """
    Chuyển câu tiếng Anh (native) → danh sách phoneme.
    Các từ phân tách bằng '|' (EN_SEP theo paper).
    """
    words = text.strip().lower().split()
    all_phonemes = []
    for i, word in enumerate(words):
        word_clean = re.sub(r"[^\w]", "", word)
        if not word_clean:
            continue
        ph = en_word_to_phoneme(word_clean)
        all_phonemes += ph
        if i < len(words) - 1:
            all_phonemes.append("|")  # ranh giới từ tiếng Anh
    return all_phonemes


# ═══════════════════════════════════════════════════════════════════════════
#  3. PHẦN VIETLISH: EN WORD → CÁCH ĐỌC TIẾNG VIỆT → PHONEME
#  Tra VIETLISH_MAP, nếu không có thì cố g2p → vn-style fallback
# ═══════════════════════════════════════════════════════════════════════════

from phoneme_set import VIETLISH_MAP

def vietlish_word_to_phoneme(word: str) -> list:
    """
    Chuyển từ tiếng Anh theo cách đọc Vietlish.
    Tra VIETLISH_MAP trước. Nếu không tìm thấy, fallback sang phiên âm native.
    """
    word = word.lower().strip()
    if word in VIETLISH_MAP:
        _, phonemes = VIETLISH_MAP[word]
        return list(phonemes)

    # Fallback: dùng phoneme tiếng Anh native
    return en_word_to_phoneme(word)


def vietlish_text_to_phoneme(text: str) -> list:
    """
    Chuyển câu Vietlish (EN words đọc kiểu VN, hoặc câu hỗn hợp VN+EN).
    Mỗi từ được tự động phân loại:
      - Từ có ký tự VN đặc trưng (đ, ă, ơ...) → xử lý bằng VN rule
      - Từ ASCII thuần → tra Vietlish map hoặc fallback EN G2P
    Phân tách bằng '$'.
    """
    words = text.strip().split()
    all_phonemes = []
    for i, word in enumerate(words):
        word_clean = re.sub(r"^[^\w\u00C0-\u1EF9]+|[^\w\u00C0-\u1EF9]+$", "", word)
        if not word_clean:
            continue
        # Nếu từ chứa ký tự VN đặc trưng → xử lý bằng VN phoneme rule
        if is_vietnamese_word(word_clean):
            ph = vn_syllable_to_phoneme(word_clean)
        else:
            ph = vietlish_word_to_phoneme(word_clean.lower())
        all_phonemes += ph
        if i < len(words) - 1:
            all_phonemes.append("$")
    return all_phonemes


# ═══════════════════════════════════════════════════════════════════════════
#  4. PHẦN IEV (CODE-SWITCHING): TOKENIZE + XỬ LÝ TỪNG NGÔN NGỮ
#  Tự động phát hiện từ tiếng Anh vs tiếng Việt
# ═══════════════════════════════════════════════════════════════════════════

# Tập hợp ký tự tiếng Việt đặc trưng (có dấu phụ âm / thanh điệu)
VN_CHARS = set("àáảãạăắặẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
               "ÀÁẢÃẠĂẮẶẲẴẬÂẦẤẨẪẪÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ")

# Từ tiếng Việt thuần ASCII phổ biến (không có dấu đặc trưng)
# Dùng để phân biệt VN vs EN trong detect_language
VN_COMMON_WORDS = {
    "tôi", "toi", "em", "anh", "chị", "chi", "bạn", "ban", "họ", "ho",
    "mình", "minh", "chúng", "chung", "các", "cac", "và", "va", "là", "la",
    "có", "co", "không", "khong", "được", "duoc", "với", "voi", "cho", "của",
    "cua", "trong", "ở", "o", "từ", "tu", "khi", "thì", "thi", "mà", "ma",
    "để", "de", "hay", "hoặc", "hoac", "nhưng", "nhung", "vì", "vi", "nếu",
    "neu", "đã", "da", "sẽ", "se", "đang", "dang", "nhà", "nha", "xe",
    "ngay", "nay", "nho", "thi", "an", "ăn", "di", "ve",
}

def is_vietnamese_word(word: str) -> bool:
    """Kiểm tra từ có phải tiếng Việt không.
    Ưu tiên: ký tự đặc trưng VN, sau đó common words list.
    """
    word_lower = word.lower()
    # Có ký tự đặc trưng tiếng Việt
    if any(ch in VN_CHARS for ch in word_lower):
        return True
    # Từ phổ biến tiếng Việt không dấu
    if word_lower in VN_COMMON_WORDS:
        return True
    return False

def is_english_word(word: str) -> bool:
    """Kiểm tra từ có phải tiếng Anh không (chỉ gồm ký tự ASCII Latin, không phải VN common)."""
    if not re.match(r"^[a-zA-Z]+$", word):
        return False
    # Không phải từ tiếng Việt thông dụng
    return word.lower() not in VN_COMMON_WORDS

def iev_text_to_phoneme(text: str) -> list:
    """
    Chuyển câu IEV (code-switching: xen kẽ VN + EN) → danh sách phoneme.

    Thuật toán:
    1. Tokenize theo khoảng trắng
    2. Mỗi từ: phát hiện ngôn ngữ (VN / EN)
    3. EN trong câu VN → dùng Vietlish (theo paper họ map sang VN pronunciation)
    4. VN → vn_syllable_to_phoneme
    5. Phân cách bằng '$'

    Ví dụ input:  "anh đang dùng laptop ở nhà"
    Từng từ:       VN    VN    VN      EN      VN   VN
    """
    words = text.strip().split()
    all_phonemes = []

    for i, word in enumerate(words):
        word_clean = re.sub(r"^[^\w\u00C0-\u1EF9]+|[^\w\u00C0-\u1EF9]+$", "", word)
        if not word_clean:
            continue

        if is_vietnamese_word(word_clean):
            # Từ tiếng Việt
            ph = vn_syllable_to_phoneme(word_clean)
        elif is_english_word(word_clean):
            # Từ tiếng Anh trong câu Việt → Vietlish pronunciation
            ph = vietlish_word_to_phoneme(word_clean.lower())
        else:
            # Fallback
            ph = vn_syllable_to_phoneme(word_clean)

        all_phonemes += ph
        if i < len(words) - 1:
            all_phonemes.append("$")

    return all_phonemes


# ═══════════════════════════════════════════════════════════════════════════
#  5. HÀM CHÍNH: AUTO-DETECT NGÔN NGỮ & CHUYỂN ĐỔI
# ═══════════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """
    Phát hiện ngôn ngữ chính của câu:
    - "vi"       : tiếng Việt thuần
    - "en"       : tiếng Anh thuần
    - "vietlish" : từ tiếng Anh đọc kiểu Việt
    - "iev"      : câu code-switching xen kẽ
    """
    words = text.strip().split()
    vn_count = sum(1 for w in words if is_vietnamese_word(w))
    en_count = sum(1 for w in words if is_english_word(w))
    total    = len(words)

    if total == 0:
        return "vi"

    vn_ratio = vn_count / total
    en_ratio = en_count / total

    if vn_ratio >= 0.8:
        return "vi"
    elif en_ratio >= 0.8:
        # Kiểm tra xem có trong Vietlish map không
        vietlish_hits = sum(1 for w in words if w.lower() in VIETLISH_MAP)
        if vietlish_hits > 0:
            return "vietlish"
        return "en"
    else:
        return "iev"


def text_to_phoneme(text: str, mode: str = "auto") -> list:
    """
    Hàm CHÍNH: chuyển text → danh sách phoneme.

    Args:
        text: Văn bản đầu vào
        mode: "auto" | "vi" | "en" | "vietlish" | "iev"

    Returns:
        Danh sách phoneme (list of str)

    Ví dụ:
        text_to_phoneme("xin chào")    → ["s","i","n","-1","$","tʃ","a","-2"]
        text_to_phoneme("inbox")       → ["ɪ","n","$","b","o","-4","k"]  (Vietlish)
        text_to_phoneme("hello world") → ["h","ə","l","oʊ","-1","|","w","ɜː","l","d","-1"]
    """
    text = text.strip()
    if not text:
        return []

    if mode == "auto":
        mode = detect_language(text)

    if mode == "vi":
        return vn_text_to_phoneme(text)
    elif mode == "en":
        return en_text_to_phoneme(text)
    elif mode == "vietlish":
        return vietlish_text_to_phoneme(text)
    elif mode == "iev":
        return iev_text_to_phoneme(text)
    else:
        return vn_text_to_phoneme(text)


# ═══════════════════════════════════════════════════════════════════════════
#  6. DEMO / MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("   TEXT → PHONEME  (Theo phương pháp của paper)")
    print("=" * 65)

    test_cases = [
        # (text, mode, mô tả)
        ("xin chào",                    "vi",       "Tiếng Việt thuần"),
        ("tôi đang đi học",             "vi",       "Câu VN đơn giản"),
        ("chúng tôi sẽ kiểm tra",       "vi",       "Câu VN phức tạp"),
        ("hello world",                 "en",       "Tiếng Anh native"),
        ("message inbox coffee",        "en",       "EN native words"),
        ("inbox",                       "vietlish", "Vietlish (in bóc)"),
        ("message coffee",              "vietlish", "Vietlish (mét xịt, cà phê)"),
        ("anh đang dùng laptop ở nhà",  "iev",      "IEV (code-switching)"),
        ("mình sẽ check email ngay",    "iev",      "IEV (VN + EN)"),
        ("tôi đang đi học",             "auto",     "Auto-detect → VN"),
        ("inbox laptop coffee",         "auto",     "Auto-detect → Vietlish"),
        ("em đang share file cho anh",  "auto",     "Auto-detect → IEV"),
    ]

    for text, mode, desc in test_cases:
        result = text_to_phoneme(text, mode=mode)
        detected = detect_language(text) if mode == "auto" else mode
        print(f"\n[{desc}]")
        print(f"  Input  : \"{text}\"")
        print(f"  Mode   : {detected}")
        print(f"  Phoneme: {result}")

    print("\n" + "=" * 65)
    print("  Sử dụng trong code:")
    print("    from text2phoneme import text_to_phoneme")
    print("    ph = text_to_phoneme('xin chào', mode='vi')")
    print("    ph = text_to_phoneme('inbox', mode='vietlish')")
    print("    ph = text_to_phoneme('anh check email', mode='iev')")
    print("=" * 65)
