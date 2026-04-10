"""
generate_dataset.py
Sinh 10,000 sample câu Việt / Anh / IEV với phoneme annotation.
Chạy: python generate_dataset.py
Output: corpus_10000.py  (dict + helper) và corpus_10000.json
"""

import random
import json
from itertools import product

random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════
#  BẢNG ÂM VỊ GỐC
# ═══════════════════════════════════════════════════════════════════════════

# Tone markers: -1 ngang, -2 huyền, -3 hỏi, -4 sắc, -5 nặng, -6 ngã
# $ = ranh giới âm tiết VN,  | = ranh giới EN

# ─── Từ điển tiếng Việt ────────────────────────────────────────────────────
#  (text_token, [syllables], [phonemes])
VN_WORDS = {
    # Đại từ / chủ ngữ
    "tôi":       (["tôi"],          ["t","o","-1","j"]),
    "mình":      (["mình"],         ["m","i","-6","ŋ"]),
    "bạn":       (["bạn"],          ["b","a","-5","n"]),
    "anh":       (["anh"],          ["a","-1","ŋ"]),
    "chị":       (["chị"],          ["tʃ","i","-5"]),
    "em":        (["em"],           ["ɛ","-1","m"]),
    "họ":        (["họ"],           ["h","ɔ","-5"]),
    "chúng tôi": (["chúng","tôi"],  ["tʃ","u","-4","ŋ","$","t","o","-1","j"]),
    "các bạn":   (["các","bạn"],    ["k","a","-4","k","$","b","a","-5","n"]),

    # Động từ
    "đang":      (["đang"],         ["d","a","-1","ŋ"]),
    "sẽ":        (["sẽ"],           ["s","ɛ","-6"]),
    "đã":        (["đã"],           ["d","a","-6"]),
    "làm":       (["làm"],          ["l","a","-2","m"]),
    "học":       (["học"],          ["h","ɔ","-5","k"]),
    "làm việc":  (["làm","việc"],   ["l","a","-2","m","$","v","i","-5","k"]),
    "gửi":       (["gửi"],          ["g","ɯ","-3","j"]),
    "nhận":      (["nhận"],         ["ŋ","ɛ","-5","n"]),
    "xem":       (["xem"],          ["s","ɛ","-1","m"]),
    "nghe":      (["nghe"],         ["ŋ","ɛ","-1"]),
    "nói":       (["nói"],          ["n","o","-4","j"]),
    "viết":      (["viết"],         ["v","i","-4","t"]),
    "đọc":       (["đọc"],          ["d","ɔ","-5","k"]),
    "mở":        (["mở"],           ["m","ɤ","-3"]),
    "tắt":       (["tắt"],          ["t","a","-4","t"]),
    "bật":       (["bật"],          ["b","a","-5","t"]),
    "chia sẻ":   (["chia","sẻ"],    ["tʃ","i","a","-1","$","s","ɛ","-3"]),
    "tải":       (["tải"],          ["t","a","-3","j"]),
    "cài":       (["cài"],          ["k","a","-2","j"]),
    "dùng":      (["dùng"],         ["z","u","-2","ŋ"]),
    "thích":     (["thích"],        ["tʰ","i","-4","k"]),
    "muốn":      (["muốn"],         ["m","u","-4","n"]),
    "cần":       (["cần"],          ["k","ɤ","-2","n"]),
    "có":        (["có"],           ["k","o","-4"]),
    "không":     (["không"],        ["x","o","-1","ŋ"]),
    "biết":      (["biết"],         ["b","i","-4","t"]),
    "hiểu":      (["hiểu"],         ["h","i","-3","w"]),
    "giúp":      (["giúp"],         ["z","u","-4","p"]),
    "gặp":       (["gặp"],          ["g","a","-5","p"]),
    "đến":       (["đến"],          ["d","e","-4","n"]),
    "về":        (["về"],           ["v","e","-2"]),
    "đi":        (["đi"],           ["d","i","-1"]),
    "ăn":        (["ăn"],           ["a","-1","n"]),
    "uống":      (["uống"],         ["u","-4","ŋ"]),
    "mua":       (["mua"],          ["m","u","a","-1"]),
    "bán":       (["bán"],          ["b","a","-4","n"]),
    "tìm":       (["tìm"],          ["t","i","-2","m"]),
    "kiểm tra":  (["kiểm","tra"],   ["k","i","-3","m","$","tʃ","a","-1"]),

    # Tính từ / trạng từ
    "tốt":       (["tốt"],          ["t","o","-4","t"]),
    "xấu":       (["xấu"],          ["s","a","-4","w"]),
    "nhanh":     (["nhanh"],        ["ɲ","a","-1","ŋ"]),
    "chậm":      (["chậm"],         ["tʃ","a","-5","m"]),
    "nhiều":     (["nhiều"],        ["ɲ","i","-2","w"]),
    "ít":        (["ít"],           ["i","-4","t"]),
    "mới":       (["mới"],          ["m","ɤ","-4","j"]),
    "cũ":        (["cũ"],           ["k","u","-6"]),
    "hay":       (["hay"],          ["h","a","-1","j"]),
    "khó":       (["khó"],          ["x","o","-4"]),
    "dễ":        (["dễ"],           ["z","ɛ","-6"]),
    "quan trọng":(["quan","trọng"], ["k","u","a","n","-1","$","tʃ","ɔ","-5","ŋ"]),
    "nhanh lên": (["nhanh","lên"],  ["ɲ","a","-1","ŋ","$","l","e","-1","n"]),
    "rất":       (["rất"],          ["r","a","-5","t"]),
    "quá":       (["quá"],          ["k","u","a","-4"]),
    "lắm":       (["lắm"],          ["l","a","-4","m"]),
    "hôm nay":   (["hôm","nay"],    ["h","o","-1","m","$","n","a","-1","j"]),
    "ngày mai":  (["ngày","mai"],   ["ŋ","a","-2","j","$","m","a","-1","j"]),
    "bây giờ":   (["bây","giờ"],    ["b","a","-1","j","$","z","ɤ","-2"]),

    # Danh từ thông dụng
    "điện thoại":(["điện","thoại"],["d","i","-5","n","$","tʰ","u","a","-5","j"]),
    "máy tính":  (["máy","tính"],  ["m","a","-4","j","$","t","i","-4","ŋ"]),
    "mạng":      (["mạng"],         ["m","a","-5","ŋ"]),
    "ứng dụng":  (["ứng","dụng"],  ["ɯ","-4","ŋ","$","z","u","-5","ŋ"]),
    "tin nhắn":  (["tin","nhắn"],  ["t","i","-1","n","$","ɲ","a","-4","n"]),
    "cuộc gọi":  (["cuộc","gọi"], ["k","u","-5","k","$","g","ɔ","-5","j"]),
    "tài khoản": (["tài","khoản"],["t","a","-2","j","$","x","u","a","-3","n"]),
    "mật khẩu":  (["mật","khẩu"], ["m","a","-5","t","$","x","a","-3","w"]),
    "file":      (["phai"],         ["f","a","j","-1"]),
    "dữ liệu":   (["dữ","liệu"],  ["z","ɯ","-6","$","l","i","-5","w"]),
    "hệ thống":  (["hệ","thống"], ["h","e","-5","$","tʰ","o","-4","ŋ"]),
    "phần mềm":  (["phần","mềm"], ["f","ɤ","-2","n","$","m","e","-2","m"]),
    "trình duyệt":(["trình","duyệt"],["tʃ","i","-2","ŋ","$","z","u","i","-5","t"]),
    "công việc": (["công","việc"], ["k","o","-1","ŋ","$","v","i","-5","k"]),
    "bạn bè":    (["bạn","bè"],    ["b","a","-5","n","$","b","ɛ","-2"]),
    "gia đình":  (["gia","đình"],  ["z","i","a","-1","$","d","i","-2","ŋ"]),
    "trường":    (["trường"],       ["tʃ","ɯ","-2","ŋ"]),
    "công ty":   (["công","ty"],   ["k","o","-1","ŋ","$","t","i","-1"]),
    "nhà":       (["nhà"],          ["ɲ","a","-2"]),
    "xe":        (["xe"],           ["s","ɛ","-1"]),
    "tiền":      (["tiền"],         ["t","i","-2","n"]),
    "thời gian": (["thời","gian"], ["tʰ","ɤ","-2","j","$","z","i","a","-1","n"]),

    # Giới từ / liên từ / cảm thán
    "và":        (["và"],           ["v","a","-2"]),
    "nhưng":     (["nhưng"],        ["ɲ","ɯ","-1","ŋ"]),
    "hoặc":      (["hoặc"],         ["h","u","a","-5","k"]),
    "vì":        (["vì"],           ["v","i","-2"]),
    "nên":       (["nên"],          ["n","e","-1","n"]),
    "để":        (["để"],           ["d","ɛ","-3"]),
    "với":       (["với"],          ["v","ɤ","-4","j"]),
    "cho":       (["cho"],          ["tʃ","ɔ","-1"]),
    "ở":         (["ở"],            ["ɤ","-3"]),
    "tại":       (["tại"],          ["t","a","-5","j"]),
    "từ":        (["từ"],           ["t","ɯ","-2"]),
    "ơi":        (["ơi"],           ["ɤ","-1","j"]),
    "ừ":         (["ừ"],            ["ɯ","-2"]),
    "vâng":      (["vâng"],         ["v","a","-1","ŋ"]),
    "dạ":        (["dạ"],           ["z","a","-5"]),
    "thôi":      (["thôi"],         ["tʰ","o","-1","j"]),
    "xong":      (["xong"],         ["s","ɔ","-1","ŋ"]),
    "rồi":       (["rồi"],          ["r","o","-2","j"]),
    "à":         (["à"],            ["a","-2"]),
    "ạ":         (["ạ"],            ["a","-5"]),
    "nhé":       (["nhé"],          ["ɲ","ɛ","-4"]),
    "nha":       (["nha"],          ["ɲ","a","-1"]),
    "đó":        (["đó"],           ["d","ɔ","-4"]),
    "này":       (["này"],          ["n","a","-2","j"]),
    "kia":       (["kia"],          ["k","i","a","-1"]),
    "mà":        (["mà"],           ["m","a","-2"]),
    "thì":       (["thì"],          ["tʰ","i","-2"]),
}

# ─── Từ điển tiếng Anh (Vietlish + IPA) ──────────────────────────────────
EN_WORDS = {
    "message":    (["mét","xịt"],         ["m","e","-4","tz","$","s","i","-5","tz"]),
    "inbox":      (["in","bóc"],          ["ɪ","n","$","b","o","-4","k"]),
    "coffee":     (["cà","phê"],          ["k","a","-2","$","f","e","-1"]),
    "laptop":     (["lép","tóp"],         ["l","e","-5","$","t","o","-4"]),
    "online":     (["on","lai"],          ["o","n","$","l","a","j"]),
    "meeting":    (["mít","tinh"],        ["m","i","-5","$","t","i","ŋ"]),
    "email":      (["i","meo"],           ["i","$","m","ɛ","w"]),
    "video":      (["vi","đê","ô"],       ["v","i","$","d","e","-1","$","o","-1"]),
    "check":      (["chéc"],             ["tʃ","e","-4","k"]),
    "share":      (["sẹ"],               ["ʃ","ɛ","-5"]),
    "like":       (["lai"],              ["l","a","j"]),
    "download":   (["đao","lôt"],        ["d","a","w","$","l","o","-4","t"]),
    "upload":     (["ắp","lôt"],         ["a","-4","p","$","l","o","-4","t"]),
    "phone":      (["phôn"],             ["f","o","n","-1"]),
    "wifi":       (["oai","phai"],       ["w","a","j","$","f","a","j"]),
    "app":        (["ép"],               ["a","-4","p"]),
    "call":       (["col"],              ["k","ɔ","l","-1"]),
    "chat":       (["chét"],             ["tʃ","ɛ","-4","t"]),
    "group":      (["gờ-rúp"],           ["g","ɯ","$","r","u","-4","p"]),
    "link":       (["lính"],             ["l","i","-4","ŋ"]),
    "post":       (["pốt"],              ["p","o","-4","t"]),
    "comment":    (["com","men"],        ["k","ɔ","m","-1","$","m","ɛ","n","-1"]),
    "update":     (["ắp","đết"],         ["a","-4","p","$","d","ɛ","-4","t"]),
    "follow":     (["fo","lô"],          ["f","ɔ","-1","$","l","o","-1"]),
    "block":      (["blóc"],             ["b","l","o","-4","k"]),
    "report":     (["ri","pót"],         ["r","i","-1","$","p","o","-4","t"]),
    "spam":       (["xpém"],             ["s","p","ɛ","-4","m"]),
    "backup":     (["béc","ắp"],         ["b","ɛ","-4","k","$","a","-4","p"]),
    "password":   (["pát","xợt"],        ["p","a","-4","t","$","s","ɤ","-5","t"]),
    "account":    (["ờ","cao"],          ["ɤ","-1","$","k","a","w","-1"]),
    "login":      (["lóc","gin"],        ["l","o","-4","k","$","z","i","n","-1"]),
    "logout":     (["lóc","ao"],         ["l","o","-4","k","$","a","w","-1"]),
    "delete":     (["đi","lít"],         ["d","i","-1","$","l","i","-4","t"]),
    "edit":       (["é","đit"],          ["ɛ","-4","$","d","i","t","-1"]),
    "cancel":     (["can","xen"],        ["k","a","n","-1","$","s","ɛ","n","-1"]),
    "confirm":    (["con","phơm"],       ["k","ɔ","n","-1","$","f","ɤ","m","-1"]),
    "submit":     (["xớp","mít"],        ["s","ɤ","-4","p","$","m","i","-4","t"]),
    "search":     (["xớc"],              ["s","ɤ","-4","k"]),
    "filter":     (["phin","tờ"],        ["f","i","n","-1","$","t","ɤ","-1"]),
    "sort":       (["sót"],              ["s","o","-4","t"]),
    "zoom":       (["zum"],              ["z","u","m","-1"]),
    "google":     (["gù","gồ"],          ["g","u","-2","$","g","o","-2"]),
    "facebook":   (["phết","búc"],       ["f","ɛ","-4","t","$","b","u","-4","k"]),
    "youtube":    (["du","tíu"],         ["z","u","-1","$","t","i","-4","w"]),
    "tiktok":     (["tích","tóc"],       ["t","i","-4","k","$","t","o","-4","k"]),
    "instagram":  (["in","xta","grem"], ["i","n","$","s","t","a","-1","$","g","r","ɛ","m","-1"]),
    "twitter":    (["tuýt","tờ"],        ["t","u","-4","j","t","$","t","ɤ","-1"]),
    "zalo":       (["da","lô"],          ["z","a","-1","$","l","o","-1"]),
    "telegram":   (["te","lê","grem"],   ["t","ɛ","-1","$","l","e","-1","$","g","r","ɛ","m","-1"]),
    "zoom call":  (["zum","col"],        ["z","u","m","-1","$","k","ɔ","l","-1"]),
    "deadline":   (["đét","lai"],        ["d","ɛ","-4","t","$","l","a","j","-1"]),
    "project":    (["pro","ject"],       ["p","r","o","-1","$","dʒ","ɛ","k","t","-1"]),
    "team":       (["tim"],              ["t","i","m","-1"]),
    "boss":       (["bốt"],              ["b","o","-4","t"]),
    "office":     (["o","phít"],         ["o","-1","$","f","i","-4","t"]),
    "work":       (["uốc"],              ["u","-4","k"]),
    "task":       (["téc"],              ["t","ɛ","-4","k"]),
    "bug":        (["bắc"],              ["b","a","-4","k"]),
    "fix":        (["phích"],            ["f","i","-4","k"]),
    "code":       (["côt"],              ["k","o","-4","t"]),
    "server":     (["sờ","vờ"],          ["s","ɤ","-1","$","v","ɤ","-1"]),
    "database":   (["đết","ba","dơ"],   ["d","ɛ","-4","t","$","b","a","-1","$","z","ɤ","-1"]),
    "deploy":     (["đi","ploi"],        ["d","i","-1","$","p","l","ɔ","j","-1"]),
    "review":     (["ri","viu"],         ["r","i","-1","$","v","i","u","-1"]),
    "feedback":   (["phít","béc"],       ["f","i","-4","t","$","b","ɛ","-4","k"]),
    "meeting room":(["mít","tinh","rum"],["m","i","-5","$","t","i","ŋ","$","r","u","m","-1"]),
    "presentation":(["pre","sen","tây","shần"],["p","r","ɛ","-1","$","s","ɛ","n","-1","$","t","a","-1","j","$","ʃ","ɤ","n","-1"]),
    "powerpoint":  (["pa","ơ","poin"],   ["p","a","-1","$","ɤ","-1","$","p","ɔ","j","n","-1"]),
    "excel":       (["ếch","xen"],       ["ɛ","-4","k","$","s","ɛ","n","-1"]),
    "word":        (["uốt"],             ["u","-4","t"]),
    "pdf":         (["pê","đê","ép"],    ["p","e","-1","$","d","e","-1","$","ɛ","-4","p"]),
    "screenshot":  (["xc","rít","shot"], ["s","k","$","r","i","-4","t","$","ʃ","ɔ","t","-1"]),
    "notification":(["no","ti","phi","kây","shần"],["n","o","-1","$","t","i","-1","$","f","i","-1","$","k","a","j","-1","$","ʃ","ɤ","n","-1"]),
    "settings":    (["xết","tinh"],      ["s","ɛ","-4","t","$","t","i","ŋ","-1"]),
    "camera":      (["ca","mê","ra"],    ["k","a","-1","$","m","e","-1","$","r","a","-1"]),
    "bluetooth":   (["blú","tút"],       ["b","l","u","-4","$","t","u","-4","t"]),
    "headphone":   (["hét","phôn"],      ["h","ɛ","-4","t","$","f","o","n","-1"]),
    "speaker":     (["xpí","kờ"],        ["s","p","i","-4","$","k","ɤ","-1"]),
    "keyboard":    (["ki","bo"],         ["k","i","-1","$","b","o","-1"]),
    "mouse":       (["mao"],             ["m","a","w","-1"]),
    "screen":      (["xc","rin"],        ["s","k","$","r","i","n","-1"]),
    "battery":     (["be","tơ","ri"],    ["b","ɛ","-1","$","t","ɤ","-1","$","r","i","-1"]),
    "charger":     (["chạc","gờ"],       ["tʃ","a","-5","k","$","g","ɤ","-1"]),
    "cable":       (["kê","bồ"],         ["k","e","-1","$","b","o","-2"]),
    "adapter":     (["a","đép","tờ"],    ["a","-1","$","d","ɛ","-4","p","$","t","ɤ","-1"]),
    "internet":    (["in","tờ","nét"],   ["i","n","$","t","ɤ","-1","$","n","ɛ","-4","t"]),
    "website":     (["uép","xai"],       ["u","-4","p","$","s","a","j","-1"]),
    "browser":     (["brao","dờ"],       ["b","r","a","w","-1","$","z","ɤ","-1"]),
    "tab":         (["téb"],             ["t","ɛ","-4","b"]),
    "folder":      (["phon","đờ"],       ["f","ɔ","n","-1","$","d","ɤ","-1"]),
    "icon":        (["ai","con"],        ["a","j","-1","$","k","ɔ","n","-1"]),
    "menu":        (["mê","niu"],        ["m","e","-1","$","ɲ","i","u","-1"]),
    "interface":   (["in","tờ","phết"],  ["i","n","$","t","ɤ","-1","$","f","ɛ","-4","t"]),
    "feature":     (["phi","chờ"],       ["f","i","-1","$","tʃ","ɤ","-1"]),
    "version":     (["vờ","sinh"],       ["v","ɤ","-1","$","ʃ","i","ŋ","-1"]),
    "install":     (["in","xto"],        ["i","n","$","s","t","o","-1"]),
    "plugin":      (["plag","in"],       ["p","l","a","k","-1","$","i","n","-1"]),
    "startup":     (["xtát","ắp"],       ["s","t","a","-4","t","$","a","-4","p"]),
    "sale":        (["xên"],             ["s","ɛ","n","-1"]),
    "order":       (["o","đờ"],          ["o","-1","$","d","ɤ","-1"]),
    "ship":        (["xíp"],             ["s","i","-4","p"]),
    "voucher":     (["vao","chờ"],       ["v","a","w","-1","$","tʃ","ɤ","-1"]),
    "trend":       (["tren"],            ["t","r","ɛ","n","-1"]),
    "event":       (["i","ven"],         ["i","-1","$","v","ɛ","n","-1"]),
    "party":       (["pa","ti"],         ["p","a","-1","$","t","i","-1"]),
    "game":        (["gêm"],             ["g","e","m","-1"]),
    "level":       (["le","vồ"],         ["l","ɛ","-1","$","v","o","-2"]),
    "stream":      (["xtrim"],           ["s","t","r","i","m","-1"]),
    "content":     (["con","ten"],       ["k","ɔ","n","-1","$","t","ɛ","n","-1"]),
    "channel":     (["chen","nồ"],       ["tʃ","ɛ","n","-1","$","n","o","-2"]),
    "subscribe":   (["xớp","xcrai"],     ["s","ɤ","-4","p","$","s","k","r","a","j","-1"]),
    "playlist":    (["plây","lít"],      ["p","l","a","j","-1","$","l","i","-4","t"]),
    "repost":      (["ri","pốt"],        ["r","i","-1","$","p","o","-4","t"]),
    "story":       (["xto","ri"],        ["s","t","o","-1","$","r","i","-1"]),
    "reel":        (["rin"],             ["r","i","n","-1"]),
    "live":        (["lai"],             ["l","a","j","-1"]),
}

# ─── Các mẫu câu ──────────────────────────────────────────────────────────
# Dạng: list of (template_fn, type)
# type: "vi" | "en" | "iev"

def _join(parts):
    """Ghép syllables và phonemes từ nhiều từ."""
    syl_all = []
    pho_all = []
    for syl, pho in parts:
        syl_all.extend(syl)
        pho_all.extend(pho)
        pho_all.append("$")   # ranh giới giữa các từ
    if pho_all and pho_all[-1] == "$":
        pho_all.pop()
    return syl_all, pho_all


def lookup(word, bank):
    return bank[word]


def w(word, bank=None):
    if bank is None:
        bank = VN_WORDS
    return bank[word]


# ═══════════════════════════════════════════════════════════════════════════
#  TEMPLATE GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

def gen_vi_simple():
    """Câu tiếng Việt đơn giản: chủ ngữ + động từ + bổ ngữ — nhiều biến thể"""
    subjects  = ["tôi","mình","bạn","anh","chị","em","họ","các bạn","chúng tôi"]
    aspects   = ["đang","sẽ","đã","vừa"]
    actions   = ["làm","học","xem","nghe","nói","viết","đọc","đi","ăn","uống",
                 "tìm","mua","tải","dùng","gửi","nhận","mở","tắt","bật","giúp",
                 "gặp","đến","về","bán","tìm","kiểm tra","chia sẻ","cài","hiểu","biết"]
    objects_vn= ["nhiều","ít","tốt","nhanh","hay","khó","dễ","mới","cũ","quan trọng",
                 "rất","quá","lắm","nhanh lên"]
    nouns     = ["điện thoại","máy tính","mạng","ứng dụng","tin nhắn","tài khoản",
                 "mật khẩu","dữ liệu","hệ thống","phần mềm","công việc","bạn bè",
                 "gia đình","nhà","xe","tiền","thời gian","trường","công ty"]
    endings   = ["nhé","nha","rồi","xong","à","ạ","thôi","đó","mà"]

    # 4 cấu trúc câu khác nhau để tăng diversity
    pattern = random.randint(0, 3)
    if pattern == 0:
        subj = random.choice(subjects); asp = random.choice(aspects)
        act  = random.choice(actions);  obj = random.choice(objects_vn)
        end  = random.choice(endings)
        text  = f"{subj} {asp} {act} {obj} {end}"
        parts = [w(subj), w(asp), w(act), w(obj), w(end)]
    elif pattern == 1:
        subj = random.choice(subjects); asp = random.choice(aspects)
        act  = random.choice(actions);  noun = random.choice(nouns)
        end  = random.choice(endings)
        text  = f"{subj} {asp} {act} {noun} {end}"
        parts = [w(subj), w(asp), w(act), w(noun), w(end)]
    elif pattern == 2:
        subj = random.choice(subjects); conj = random.choice(["và","nhưng","hoặc"])
        subj2 = random.choice(subjects)
        asp  = random.choice(aspects);  act = random.choice(actions)
        end  = random.choice(endings)
        text  = f"{subj} {conj} {subj2} {asp} {act} {end}"
        parts = [w(subj), w(conj), w(subj2), w(asp), w(act), w(end)]
    else:
        subj = random.choice(subjects); asp = random.choice(aspects)
        act  = random.choice(actions);  prep = random.choice(["cho","với","ở","từ"])
        noun = random.choice(nouns);    end = random.choice(endings)
        text  = f"{subj} {asp} {act} {prep} {noun} {end}"
        parts = [w(subj), w(asp), w(act), w(prep), w(noun), w(end)]

    syl, pho = _join(parts)
    return text, syl, pho


def gen_vi_question():
    """Câu hỏi tiếng Việt — nhiều mẫu mở rộng"""
    subjects  = ["bạn","anh","chị","em","tôi","họ"]
    modals    = ["có","không","muốn","cần","thích","biết","hiểu"]
    actions   = ["làm","học","xem","đi","ăn","uống","tìm","mua","dùng","gửi",
                 "chia sẻ","tải","cài","gặp","đến","về","đọc","nghe","nói"]
    nouns     = ["điện thoại","máy tính","mạng","ứng dụng","tin nhắn","tài khoản",
                 "dữ liệu","công việc","bạn bè","gia đình","nhà","tiền","thời gian"]
    aspects   = ["đang","sẽ","đã"]
    time_words= ["hôm nay","ngày mai","bây giờ"]
    adjectives= ["tốt","hay","nhiều","ít","nhanh","khó","dễ","mới","quan trọng"]
    endings   = ["không","à","ạ","nhé","mà","đó","thôi"]

    pattern = random.randint(0, 5)
    if pattern == 0:
        subj = random.choice(subjects); modal = random.choice(modals)
        act  = random.choice(actions);  end   = random.choice(endings)
        text  = f"{subj} {modal} {act} {end}"
        parts = [w(subj), w(modal), w(act), w(end)]
    elif pattern == 1:
        subj = random.choice(subjects); asp = random.choice(aspects)
        act  = random.choice(actions);  end = random.choice(endings)
        text  = f"{subj} {asp} {act} {end}"
        parts = [w(subj), w(asp), w(act), w(end)]
    elif pattern == 2:
        subj = random.choice(subjects); modal = random.choice(["có","muốn","cần","thích"])
        noun = random.choice(nouns);    end   = random.choice(endings)
        text  = f"{subj} {modal} {noun} {end}"
        parts = [w(subj), w(modal), w(noun), w(end)]
    elif pattern == 3:
        time = random.choice(time_words); subj = random.choice(subjects)
        asp  = random.choice(aspects);    act  = random.choice(actions)
        end  = random.choice(endings)
        text  = f"{time} {subj} {asp} {act} {end}"
        parts = [w(time), w(subj), w(asp), w(act), w(end)]
    elif pattern == 4:
        subj = random.choice(subjects); asp = random.choice(aspects)
        adj  = random.choice(adjectives); adv = random.choice(["rất","quá","lắm"])
        end  = random.choice(endings)
        text  = f"{subj} {asp} {adj} {adv} {end}"
        parts = [w(subj), w(asp), w(adj), w(adv), w(end)]
    else:
        subj = random.choice(subjects); noun = random.choice(nouns)
        adj  = random.choice(adjectives); end  = random.choice(endings)
        text  = f"{noun} của {subj} {adj} {end}"
        of_syl = ["của"]; of_pho = ["k","ɯ","a","-3"]
        syl_s, pho_s = w(subj); syl_n, pho_n = w(noun)
        syl_a, pho_a = w(adj);  syl_e, pho_e = w(end)
        syl = syl_n + of_syl + syl_s + syl_a + syl_e
        pho = pho_n + ["$"] + of_pho + ["$"] + pho_s + ["$"] + pho_a + ["$"] + pho_e
        return text, syl, pho

    syl, pho = _join(parts)
    return text, syl, pho


def gen_vi_compound():
    """Câu tiếng Việt phức — mở rộng nhiều mẫu"""
    subjects  = ["tôi","mình","bạn","anh","chị","em","họ","chúng tôi","các bạn"]
    conjuncts = ["và","nhưng","hoặc","vì","nên","mà","thì"]
    actions   = ["làm việc","học","xem","tìm","kiểm tra","chia sẻ","gửi","nhận",
                 "đọc","nghe","nói","viết","tải","cài","dùng","mua","bán"]
    aspects   = ["đang","sẽ","đã","vừa"]
    adverbs   = ["rất","quá","lắm","nhiều","ít","nhanh","tốt"]
    nouns     = ["điện thoại","máy tính","mạng","ứng dụng","tin nhắn","tài khoản",
                 "dữ liệu","công việc","bạn bè","gia đình","nhà","tiền","thời gian",
                 "trường","công ty","phần mềm","hệ thống"]
    endings   = ["nhé","nha","rồi","xong","à","ạ","thôi","đó","mà"]

    pattern = random.randint(0, 4)
    if pattern == 0:
        s1 = random.choice(subjects); conj = random.choice(conjuncts)
        s2 = random.choice(subjects); asp  = random.choice(aspects)
        act = random.choice(actions); end  = random.choice(endings)
        text  = f"{s1} {conj} {s2} {asp} {act} {end}"
        parts = [w(s1), w(conj), w(s2), w(asp), w(act), w(end)]
    elif pattern == 1:
        s1  = random.choice(subjects); asp = random.choice(aspects)
        act = random.choice(actions);  adv = random.choice(adverbs)
        conj= random.choice(conjuncts); noun = random.choice(nouns)
        end = random.choice(endings)
        text  = f"{s1} {asp} {act} {adv} {conj} {noun} {end}"
        parts = [w(s1), w(asp), w(act), w(adv), w(conj), w(noun), w(end)]
    elif pattern == 2:
        s1 = random.choice(subjects); asp1 = random.choice(aspects)
        act1= random.choice(actions); conj = random.choice(conjuncts)
        asp2= random.choice(aspects); act2 = random.choice(actions)
        end = random.choice(endings)
        text  = f"{s1} {asp1} {act1} {conj} {asp2} {act2} {end}"
        parts = [w(s1), w(asp1), w(act1), w(conj), w(asp2), w(act2), w(end)]
    elif pattern == 3:
        prep= random.choice(["vì","để","nên"])
        s1  = random.choice(subjects); act1 = random.choice(actions)
        s2  = random.choice(subjects); asp  = random.choice(aspects)
        act2= random.choice(actions);  end  = random.choice(endings)
        text  = f"{prep} {s1} {act1} nên {s2} {asp} {act2} {end}"
        nên_syl = ["nên"]; nên_pho = ["n","e","-1","n"]
        syl_p, pho_p = w(prep); syl_s1, pho_s1 = w(s1)
        syl_a1, pho_a1 = w(act1); syl_s2, pho_s2 = w(s2)
        syl_asp, pho_asp = w(asp); syl_a2, pho_a2 = w(act2)
        syl_e, pho_e = w(end)
        syl = syl_p+syl_s1+syl_a1+nên_syl+syl_s2+syl_asp+syl_a2+syl_e
        pho = (pho_p+["$"]+pho_s1+["$"]+pho_a1+["$"]+nên_pho+["$"]
               +pho_s2+["$"]+pho_asp+["$"]+pho_a2+["$"]+pho_e)
        return text, syl, pho
    else:
        noun = random.choice(nouns); s1 = random.choice(subjects)
        asp  = random.choice(aspects); adv = random.choice(adverbs)
        end  = random.choice(endings)
        text  = f"{noun} của {s1} {asp} {adv} {end}"
        của_syl = ["của"]; của_pho = ["k","ɯ","a","-3"]
        syl_n,pho_n = w(noun); syl_s,pho_s = w(s1)
        syl_a,pho_a = w(asp);  syl_ad,pho_ad = w(adv)
        syl_e,pho_e = w(end)
        syl = syl_n+của_syl+syl_s+syl_a+syl_ad+syl_e
        pho = pho_n+["$"]+của_pho+["$"]+pho_s+["$"]+pho_a+["$"]+pho_ad+["$"]+pho_e
        return text, syl, pho

    syl, pho = _join(parts)
    return text, syl, pho


def gen_en_sentence():
    """Câu hoàn toàn tiếng Anh — mở rộng bằng cách tổ hợp verb+object"""
    verbs = ["check","like","download","upload","share","follow","block","report",
             "delete","edit","cancel","confirm","submit","search","filter","sort",
             "update","install","deploy","backup","fix","review","stream","subscribe",
             "repost","login","logout"]
    objects = ["email","post","video","link","app","file","account","password",
               "content","channel","playlist","settings","database","server","plugin",
               "feedback","comment","notification","tab","folder","icon","menu",
               "screenshot","feature","version","interface","tab","reel","story"]
    extras = ["google","facebook","youtube","tiktok","instagram","zalo","telegram",
              "twitter","zoom","excel","word","pdf","powerpoint","game","stream"]

    pattern = random.randint(0, 3)
    if pattern == 0:
        v = random.choice(verbs); o = random.choice(objects)
        words = [v, o]
    elif pattern == 1:
        v = random.choice(verbs); extra = random.choice(extras)
        words = [v, extra]
    elif pattern == 2:
        v1 = random.choice(verbs); o = random.choice(objects)
        v2 = random.choice(verbs)
        words = [v1, o, v2]
    else:
        v = random.choice(verbs); o1 = random.choice(objects); o2 = random.choice(objects)
        words = [v, o1, o2]

    parts = [lookup(ww, EN_WORDS) for ww in words if ww in EN_WORDS]
    if not parts:
        parts = [lookup("check", EN_WORDS), lookup("email", EN_WORDS)]
    text  = " ".join(ww for ww in words if ww in EN_WORDS)
    syl, pho = _join(parts)
    return text, syl, pho


def gen_iev_sentence():
    """Câu IEV: tiếng Việt xen tiếng Anh — 8 pattern khác nhau"""
    patterns = [
        _iev_pattern1, _iev_pattern2, _iev_pattern3, _iev_pattern4,
        _iev_pattern5, _iev_pattern6, _iev_pattern7, _iev_pattern8,
    ]
    return random.choice(patterns)()


def _iev_pattern1():
    """[subj] [aspect] [EN_action] [VN_ending]"""
    subj   = random.choice(["tôi","mình","anh","chị","em","bạn","họ"])
    aspect = random.choice(["đang","sẽ","vừa","đã"])
    en_w   = random.choice([w for w in EN_WORDS if w in
                            ["check","update","download","upload","share","like","follow",
                             "fix","review","submit","cancel","search","edit","stream",
                             "delete","backup","deploy","login","logout","report","block",
                             "sort","filter","repost","subscribe","install","confirm"]])
    ending = random.choice(["rồi","nhé","nha","xong","đó","à","ạ","thôi","mà"])
    text  = f"{subj} {aspect} {en_w} {ending}"
    syl_v, pho_v = _join([w(subj), w(aspect)])
    syl_e, pho_e = lookup(en_w, EN_WORDS)
    syl_end, pho_end = w(ending)
    syl = syl_v + syl_e + syl_end
    pho = pho_v + ["$"] + pho_e + ["$"] + pho_end
    return text, syl, pho


def _iev_pattern2():
    """[subj] [verb] [EN_obj] cho [target] [ending]"""
    subj   = random.choice(["tôi","anh","chị","em","mình","bạn"])
    verb   = random.choice(["gửi","chia sẻ","tải","xem","nhận","mở","tìm","kiểm tra"])
    en_w   = random.choice([w for w in EN_WORDS if w in
                            ["link","video","file","email","message","post","content",
                             "comment","feedback","screenshot","report","pdf","excel","word"]])
    target = random.choice(["bạn","anh","chị","em","họ","tôi","mình"])
    ending = random.choice(["nhé","nha","đó","rồi","à","ạ","thôi"])
    text = f"{subj} {verb} {en_w} cho {target} {ending}"
    syl_v, pho_v = _join([w(subj), w(verb)])
    syl_e, pho_e = lookup(en_w, EN_WORDS)
    syl_t, pho_t = _join([w("cho"), w(target), w(ending)])
    syl = syl_v + syl_e + syl_t
    pho = pho_v + ["$"] + pho_e + ["$"] + pho_t
    return text, syl, pho


def _iev_pattern3():
    """[EN_app/platform] [VN_adj] [VN_adv]"""
    en_w  = random.choice(["facebook","youtube","tiktok","instagram","zalo","telegram",
                            "google","twitter","zoom","game","app","channel","content",
                            "website","browser","instagram","reel","story","live","stream"])
    adj   = random.choice(["tốt","hay","nhanh","nhiều","mới","khó","dễ","quan trọng"])
    adv   = random.choice(["lắm","quá","rất"])
    ending= random.choice(["nhé","à","ạ","thôi","mà","đó","nha"])
    en_w_safe = en_w if en_w in EN_WORDS else "app"
    text  = f"{en_w} {adj} {adv} {ending}"
    syl_e, pho_e = lookup(en_w_safe, EN_WORDS)
    syl_v, pho_v = _join([w(adj), w(adv), w(ending)])
    syl = syl_e + syl_v
    pho = pho_e + ["$"] + pho_v
    return text, syl, pho


def _iev_pattern4():
    """[VN_question_start] [EN_thing] không"""
    starters = [
        ("bạn có dùng",  ["bạn","có","dùng"]),
        ("anh có biết",  ["anh","có","biết"]),
        ("chị thích",    ["chị","thích"]),
        ("em cần",       ["em","cần"]),
        ("tôi muốn",     ["tôi","muốn"]),
        ("mình đang dùng",["mình","đang","dùng"]),
        ("bạn đã thử",   ["bạn","đã"]),
        ("anh có mua",   ["anh","có","mua"]),
    ]
    q_text_base, q_words = random.choice(starters)
    en_w = random.choice([w for w in EN_WORDS if w in
                          ["app","game","feature","channel","playlist","tab","folder",
                           "icon","menu","settings","plugin","version","interface",
                           "website","browser","keyboard","mouse","screen","speaker",
                           "headphone","bluetooth","camera","charger","cable","adapter"]])
    ending = random.choice(["không","à","ạ","nhé","chưa"])
    text = f"{q_text_base} {en_w} {ending}"
    syl_v, pho_v = _join([w(qw) for qw in q_words if qw in VN_WORDS])
    syl_e, pho_e = lookup(en_w, EN_WORDS)
    syl_end, pho_end = w(ending)
    syl = syl_v + syl_e + syl_end
    pho = pho_v + ["$"] + pho_e + ["$"] + pho_end
    return text, syl, pho


def _iev_pattern5():
    """[VN_time] [subj] có [EN_event] [ending]"""
    times  = ["hôm nay","ngày mai","bây giờ"]
    subjs  = ["tôi","anh","chị","em","bạn","họ","mình","chúng tôi"]
    en_events = ["meeting","event","deadline","zoom call","presentation",
                 "call","live","stream","party","game","sale","order"]
    time_w = random.choice(times)
    subj   = random.choice(subjs)
    en_w   = random.choice(en_events)
    ending = random.choice(["nhé","à","ạ","rồi","đó","nha","không","thôi"])
    en_w_safe = en_w if en_w in EN_WORDS else "meeting"
    text = f"{time_w} {subj} có {en_w_safe} {ending}"
    syl_v, pho_v = _join([w(time_w), w(subj), w("có")])
    syl_e, pho_e = lookup(en_w_safe, EN_WORDS)
    syl_end, pho_end = w(ending)
    syl = syl_v + syl_e + syl_end
    pho = pho_v + ["$"] + pho_e + ["$"] + pho_end
    return text, syl, pho


def _iev_pattern6():
    """[subj] [VN_verb] [VN_noun] trên [EN_platform] [ending]"""
    subj    = random.choice(["tôi","anh","chị","em","bạn","mình","họ"])
    verb    = random.choice(["xem","chia sẻ","tìm","đăng","học","đọc","nghe","nói","viết"])
    noun    = random.choice(["video","tin nhắn","dữ liệu","công việc","bạn bè",
                             "gia đình","thời gian"])
    trên_syl= ["trên"]; trên_pho = ["tʃ","r","ɛ","-1","n"]
    platform= random.choice(["facebook","youtube","tiktok","instagram","zalo","telegram",
                              "twitter","zoom","google","website","app","channel"])
    ending  = random.choice(["nhé","à","ạ","rồi","đó","nha","thôi","mà"])
    en_w_safe = platform if platform in EN_WORDS else "app"
    text = f"{subj} {verb} {noun} trên {en_w_safe} {ending}"
    syl_vn, pho_vn = _join([w(subj), w(verb), w(noun)])
    syl_e,  pho_e  = lookup(en_w_safe, EN_WORDS)
    syl_end, pho_end = w(ending)
    syl = syl_vn + trên_syl + syl_e + syl_end
    pho = pho_vn + ["$"] + trên_pho + ["$"] + pho_e + ["$"] + pho_end
    return text, syl, pho


def _iev_pattern7():
    """[subj] [VN_need_verb] [EN_action] [VN_object] [ending]"""
    subj    = random.choice(["tôi","anh","chị","em","bạn","mình"])
    need    = random.choice(["muốn","cần","phải","nên"])
    en_act  = random.choice([w for w in EN_WORDS if w in
                             ["check","update","download","fix","review","submit",
                              "cancel","search","edit","delete","backup","deploy",
                              "install","confirm","report","block","share","like"]])
    noun    = random.choice(["điện thoại","máy tính","mạng","ứng dụng","tài khoản",
                             "mật khẩu","dữ liệu","hệ thống","phần mềm","công việc"])
    ending  = random.choice(["nhé","nha","đó","rồi","à","ạ","ngay","thôi","mà"])
    need_map = {
        "muốn": (["muốn"], ["m","u","-4","n"]),
        "cần":  (["cần"],  ["k","ɤ","-2","n"]),
        "phải": (["phải"], ["f","a","-3","j"]),
        "nên":  (["nên"],  ["n","e","-1","n"]),
    }
    en_w_safe = en_act if en_act in EN_WORDS else "check"
    text = f"{subj} {need} {en_w_safe} {noun} {ending}"
    syl_s, pho_s = w(subj)
    syl_n, pho_n = need_map[need]
    syl_e, pho_e = lookup(en_w_safe, EN_WORDS)
    syl_o, pho_o = w(noun)
    ngay_syl = ["ngay"] if ending == "ngay" else []
    ngay_pho = ["ŋ","a","-1","j"] if ending == "ngay" else []
    real_end  = "nhé" if ending == "ngay" else ending
    syl_end, pho_end = w(real_end)
    syl = syl_s + syl_n + syl_e + syl_o + ngay_syl + syl_end
    pho = pho_s+["$"]+pho_n+["$"]+pho_e+["$"]+pho_o+(["$"]+ngay_pho if ngay_pho else [])+["$"]+pho_end
    return text, syl, pho


def _iev_pattern8():
    """[VN_exclamation_context] [EN_word] [VN_emphasis]"""
    contexts = [
        ("ôi trời",  ["ôi","trời"],  [["o","-1","j"],["tʃ","ɤ","-2","j"]]),
        ("thật ra",  ["thật","ra"],  [["tʰ","a","-5","t"],["r","a","-1"]]),
        ("nói thật", ["nói","thật"], [["n","o","-4","j"],["tʰ","a","-5","t"]]),
        ("theo mình",["theo","mình"],[["tʰ","ɛ","w","-1"],["m","i","-6","ŋ"]]),
        ("hóa ra",   ["hóa","ra"],   [["h","o","a","-4"],["r","a","-1"]]),
    ]
    en_w = random.choice(["app","game","feature","content","trend","event","sale",
                           "stream","live","reel","story","channel","playlist","version",
                           "update","deadline","meeting","call","zoom"])
    emphasis = random.choice(["tốt","hay","nhiều","nhanh","mới","quan trọng","rất tốt"])
    ending   = random.choice(["nhé","à","ạ","lắm","quá","đó","mà"])
    ctx_text, ctx_syl, ctx_phos = random.choice(contexts)
    en_w_safe = en_w if en_w in EN_WORDS else "app"
    emp_word  = emphasis.split()[0]   # lấy từ đầu nếu có khoảng trắng
    emp_word2 = emphasis.split()[-1]
    text = f"{ctx_text} cái {en_w_safe} này {emp_word} {ending}"
    cái_syl = ["cái"]; cái_pho = ["k","a","-4","j"]
    này_syl = ["này"]; này_pho = ["n","a","-2","j"]
    syl_ctx = [s for ss in ctx_syl for s in [ss]]
    pho_ctx = []
    for ph in ctx_phos:
        pho_ctx += ph + ["$"]
    syl_e, pho_e = lookup(en_w_safe, EN_WORDS)
    emp_syl_val = [emp_word]; emp_pho_val = VN_WORDS.get(emp_word, ([emp_word], [emp_word]))[1]
    end_syl, end_pho = w(ending)
    syl = syl_ctx + cái_syl + syl_e + này_syl + emp_syl_val + end_syl
    pho = pho_ctx + cái_pho+["$"]+pho_e+["$"]+này_pho+["$"]+emp_pho_val+["$"]+end_pho
    return text, syl, pho


def _iev_pattern1():
    """tôi/anh/chị đang [EN_action] [VN_object]"""
    subj   = random.choice(["tôi","mình","anh","chị","em","bạn"])
    aspect = random.choice(["đang","sẽ","vừa","đã"])
    en_w   = random.choice(["check","update","download","upload","share","like","follow",
                             "fix","review","submit","cancel","search","edit","stream"])
    ending = random.choice(["rồi","nhé","nha","xong","đó","à"])
    if en_w not in EN_WORDS:
        en_w = "check"
    text  = f"{subj} {aspect} {en_w} {ending}"
    syl_v, pho_v = _join([w(subj), w(aspect)])
    syl_e, pho_e = lookup(en_w, EN_WORDS)
    syl_end, pho_end = w(ending)
    syl = syl_v + syl_e + syl_end
    pho = pho_v + ["$"] + pho_e + ["$"] + pho_end
    return text, syl, pho


def _iev_pattern2():
    """[VN_subj] [VN_verb] [EN_obj] cho [VN_target]"""
    subj   = random.choice(["tôi","anh","chị","em"])
    verb   = random.choice(["gửi","chia sẻ","tải","xem"])
    en_w   = random.choice(["link","video","file","email","message","post"])
    target = random.choice(["bạn","anh","chị","em","họ"])
    ending = random.choice(["nhé","nha","đó","rồi"])
    if en_w not in EN_WORDS:
        en_w = "link"
    text = f"{subj} {verb} {en_w} cho {target} {ending}"
    syl_v, pho_v = _join([w(subj), w(verb)])
    syl_e, pho_e = lookup(en_w, EN_WORDS)
    syl_t, pho_t = _join([w("cho"), w(target), w(ending)])
    syl = syl_v + syl_e + syl_t
    pho = pho_v + ["$"] + pho_e + ["$"] + pho_t
    return text, syl, pho


def _iev_pattern3():
    """/[EN_app] [VN_adj] lắm/quá"""
    en_w  = random.choice(["facebook","youtube","tiktok","instagram","zalo","telegram",
                            "google","twitter","zoom"])
    adj   = random.choice(["tốt","hay","nhanh","nhiều","mới"])
    adv   = random.choice(["lắm","quá","rất"])
    text  = f"{en_w} {adj} {adv}"
    syl_e, pho_e = lookup(en_w, EN_WORDS)
    syl_v, pho_v = _join([w(adj), w(adv)])
    syl = syl_e + syl_v
    pho = pho_e + ["$"] + pho_v
    return text, syl, pho


def _iev_pattern4():
    """[VN_question] [EN_feature] không"""
    question_starters = [
        ("bạn có dùng","bạn","có","dùng"),
        ("anh có biết","anh","có","biết"),
        ("chị thích","chị","thích"),
        ("em cần","em","cần"),
    ]
    q_parts_info = random.choice(question_starters)
    q_text  = q_parts_info[0]
    q_words = q_parts_info[1:]
    en_w    = random.choice(["app","game","feature","channel","playlist","tab",
                              "folder","icon","menu","settings"])
    if en_w not in EN_WORDS:
        en_w = "app"
    text = f"{q_text} {en_w} không"
    syl_v, pho_v = _join([w(qw) for qw in q_words if qw in VN_WORDS])
    syl_e, pho_e = lookup(en_w, EN_WORDS)
    syl_kh, pho_kh = w("không")
    syl = syl_v + syl_e + syl_kh
    pho = pho_v + ["$"] + pho_e + ["$"] + pho_kh
    return text, syl, pho


def _iev_pattern5():
    """[VN_time] [VN_subj] có [EN_event]"""
    times  = ["hôm nay","ngày mai","bây giờ"]
    subjs  = ["tôi","anh","chị","em","bạn","họ"]
    en_events = ["meeting","event","deadline","zoom call","presentation",
                 "call","live","stream"]
    time_w = random.choice(times)
    subj   = random.choice(subjs)
    en_w   = random.choice(en_events)
    ending = random.choice(["nhé","à","ạ","rồi","đó"])
    if en_w not in EN_WORDS:
        en_w = "meeting"
    text = f"{time_w} {subj} có {en_w} {ending}"
    syl_v, pho_v = _join([w(time_w), w(subj), w("có")])
    syl_e, pho_e = lookup(en_w, EN_WORDS)
    syl_end, pho_end = w(ending)
    syl = syl_v + syl_e + syl_end
    pho = pho_v + ["$"] + pho_e + ["$"] + pho_end
    return text, syl, pho


# ═══════════════════════════════════════════════════════════════════════════
#  SINH 10,000 SAMPLES
# ═══════════════════════════════════════════════════════════════════════════

GENERATORS = [
    (gen_vi_simple,    "vi",  2500),   # 25% tiếng Việt đơn giản
    (gen_vi_question,  "vi",  1500),   # 15% câu hỏi VN
    (gen_vi_compound,  "vi",  1500),   # 15% câu VN phức
    (gen_en_sentence,  "en",  1500),   # 15% câu tiếng Anh
    (gen_iev_sentence, "iev", 3000),   # 30% câu IEV
]


def generate_all(total: int = 10_000) -> dict:
    """
    Sinh tất cả samples.
    Returns: dict  key=text_string, value=(vietnamese_syllables, representative_phonemes)
    """
    corpus = {}
    seen   = set()
    errors = 0

    for gen_fn, lang, count in GENERATORS:
        generated = 0
        attempts  = 0
        max_attempts = count * 20

        while generated < count and attempts < max_attempts:
            attempts += 1
            try:
                result = gen_fn()
                if len(result) == 3:
                    text, syl, pho = result
                else:
                    continue

                text = text.strip()
                if not text or text in seen:
                    continue

                seen.add(text)
                corpus[text] = (syl, pho)
                generated += 1

            except (KeyError, IndexError):
                errors += 1
                continue

        print(f"  [{lang.upper():3s}] {gen_fn.__name__:25s}: {generated:5d} samples")

    print(f"\n  Tổng: {len(corpus):,} samples | Lỗi bỏ qua: {errors}")
    return corpus


# ═══════════════════════════════════════════════════════════════════════════
#  XUẤT FILE
# ═══════════════════════════════════════════════════════════════════════════

def save_as_python(corpus: dict, path: str = "corpus_10000.py"):
    """Xuất corpus dưới dạng file Python với dict và helper functions."""
    lines = [
        '"""',
        'corpus_10000.py',
        f'10,000 câu Việt/Anh/IEV với annotation âm vị.',
        'Cấu trúc: CORPUS[text] = (vietnamese_syllables, representative_phonemes)',
        '',
        'Tone markers: -1=ngang, -2=huyền, -3=hỏi, -4=sắc, -5=nặng, -6=ngã',
        'Ranh giới:    $=âm tiết VN,  |=âm tiết EN,  —=liên kết EN',
        '"""',
        '',
        'CORPUS = {',
    ]

    for text, (syl, pho) in corpus.items():
        syl_repr = repr(syl)
        pho_repr = repr(pho)
        safe_text = text.replace("'", "\\'")
        lines.append(f"    '{safe_text}': ({syl_repr}, {pho_repr}),")

    lines += [
        '}',
        '',
        '',
        '# ── Helper functions ──────────────────────────────────────────────────',
        '',
        'def get_syllables(text: str) -> list:',
        '    """Lấy danh sách âm tiết của câu."""',
        '    if text not in CORPUS:',
        '        return []',
        '    return CORPUS[text][0]',
        '',
        '',
        'def get_phonemes(text: str) -> list:',
        '    """Lấy chuỗi âm vị đại diện của câu."""',
        '    if text not in CORPUS:',
        '        return []',
        '    return CORPUS[text][1]',
        '',
        '',
        'def get_tones(text: str) -> list:',
        '    """Lấy danh sách thanh điệu trong câu."""',
        '    tone_set = {"-1","-2","-3","-4","-5","-6"}',
        '    return [p for p in get_phonemes(text) if p in tone_set]',
        '',
        '',
        'def filter_by_lang(lang: str) -> dict:',
        '    """',
        '    Lọc corpus theo ngôn ngữ.',
        '    lang: "vi" | "en" | "iev"',
        '    Phân loại dựa theo tỉ lệ âm vị tiếng Anh.',
        '    """',
        '    en_phones = {"æ","ɒ","ʌ","ɪ","ʊ","ə","iː","uː","eɪ","aɪ","oʊ",',
        '                 "dʒ","tʃ","ʒ","θ","ð","ʃ"}',
        '    result = {}',
        '    for text, (syl, pho) in CORPUS.items():',
        '        n_en = sum(1 for p in pho if p in en_phones)',
        '        ratio = n_en / max(len(pho), 1)',
        '        if lang == "en" and ratio > 0.3:',
        '            result[text] = (syl, pho)',
        '        elif lang == "vi" and ratio < 0.05:',
        '            result[text] = (syl, pho)',
        '        elif lang == "iev" and 0.05 <= ratio <= 0.3:',
        '            result[text] = (syl, pho)',
        '    return result',
        '',
        '',
        'def stats() -> dict:',
        '    """Thống kê nhanh về corpus."""',
        '    tone_map = {"-1":"ngang","-2":"huyền","-3":"hỏi",',
        '                "-4":"sắc","-5":"nặng","-6":"ngã"}',
        '    tone_counts = {name: 0 for name in tone_map.values()}',
        '    total_phones = 0',
        '    for _, (_, pho) in CORPUS.items():',
        '        total_phones += len(pho)',
        '        for p in pho:',
        '            if p in tone_map:',
        '                tone_counts[tone_map[p]] += 1',
        '    return {',
        '        "total_samples": len(CORPUS),',
        '        "total_phonemes": total_phones,',
        '        "avg_phonemes_per_sample": round(total_phones / max(len(CORPUS),1), 1),',
        '        "tone_distribution": tone_counts,',
        '    }',
        '',
        '',
        'if __name__ == "__main__":',
        '    import json',
        '    s = stats()',
        '    print(f"Corpus: {s[\'total_samples\']:,} samples")',
        '    print(f"Tổng âm vị: {s[\'total_phonemes\']:,}")',
        '    print(f"Trung bình âm vị/câu: {s[\'avg_phonemes_per_sample\']}")',
        '    print(f"Phân bố thanh điệu:")',
        '    for tone, count in s["tone_distribution"].items():',
        '        print(f"  {tone}: {count:,}")',
        '    # Ví dụ',
        '    sample_keys = list(CORPUS.keys())[:3]',
        '    print("\\nVí dụ 3 câu đầu:")',
        '    for k in sample_keys:',
        '        syl, pho = CORPUS[k]',
        '        print(f"  Câu     : {k}")',
        '        print(f"  Âm tiết : {syl}")',
        '        print(f"  Âm vị   : {pho}")',
        '        print()',
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[✓] Đã lưu: {path} ({len(corpus):,} entries)")


def save_as_json(corpus: dict, path: str = "corpus_10000.json"):
    """Xuất corpus dưới dạng JSON (dùng cho training pipeline)."""
    data = []
    for text, (syl, pho) in corpus.items():
        data.append({
            "text":                 text,
            "vietnamese_syllables": syl,
            "representative_phonemes": pho,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[✓] Đã lưu: {path} ({len(data):,} entries)")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print(" Sinh 10,000 sample câu Việt / Anh / IEV")
    print("="*60)
    print()

    corpus = generate_all(10_000)

    save_as_python(corpus, "corpus_10000.py")
    save_as_json(corpus,   "corpus_10000.json")

    # Thống kê nhanh
    print("\n── Thống kê nhanh ──────────────────────────────────────")
    tone_map   = {"-1":"ngang","-2":"huyền","-3":"hỏi","-4":"sắc","-5":"nặng","-6":"ngã"}
    tone_cnt   = {n: 0 for n in tone_map.values()}
    total_pho  = 0
    for _, (_, pho) in corpus.items():
        total_pho += len(pho)
        for p in pho:
            if p in tone_map:
                tone_cnt[tone_map[p]] += 1

    print(f"  Tổng sample    : {len(corpus):,}")
    print(f"  Tổng âm vị    : {total_pho:,}")
    print(f"  Trung bình/câu : {total_pho/max(len(corpus),1):.1f} âm vị")
    print(f"  Phân bố thanh điệu:")
    for tone, cnt in tone_cnt.items():
        bar = "█" * (cnt // 500)
        print(f"    {tone:6s}: {cnt:6,} {bar}")

    print("\n── Ví dụ 5 câu mẫu ─────────────────────────────────────")
    for i, (text, (syl, pho)) in enumerate(list(corpus.items())[:5]):
        print(f"\n  [{i+1}] Câu    : {text}")
        print(f"      Âm tiết: {syl}")
        print(f"      Âm vị  : {pho}")
