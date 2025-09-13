import os
import re
from pathlib import Path

# ======= cấu hình =======
ROOT = Path("../../data/ACDC/ACDC_training_volumes")   # đổi nếu thư mục khác
OUT_FILE = "test.txt"

# danh sách bệnh nhân cần lấy
patients = [
    "patient001","patient017","patient018","patient022","patient025",
    "patient027","patient032","patient044","patient049","patient057",
    "patient079","patient081","patient084","patient095","patient096",
]

# ======= hàm hỗ trợ =======
pat_set = set(patients)
fname_re = re.compile(r"^(patient\d{3})_frame(\d+)\.h5$", re.IGNORECASE)

def natural_key(name: str):
    """sắp xếp theo (patientID, frame số)"""
    m = fname_re.match(name)
    if not m:
        return (name, 10**9)  # đẩy các tên lạ xuống cuối
    pid, fr = m.group(1), int(m.group(2))
    return (pid, fr)

# ======= quét & lọc =======
if not ROOT.exists():
    raise FileNotFoundError(f"Không tìm thấy thư mục: {ROOT.resolve()}")

kept = set()
for fn in os.listdir(ROOT):
    if not fn.lower().endswith(".h5"):
        continue
    m = fname_re.match(fn)
    if not m:
        continue
    pid = m.group(1)
    if pid in pat_set:
        kept.add(fn)  # dùng set để loại trùng

# ======= ghi file =======
lines = sorted(kept, key=natural_key)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

print(f"Đã ghi {len(lines)} dòng vào {OUT_FILE}")
