import os
import re
from pathlib import Path

# ======= cấu hình =======
ROOT = Path("../../data/MSCMR/MSCMR_training_volumes")   # đổi nếu thư mục khác
OUT_FILE = "test.txt"

# danh sách bệnh nhân cần lấy (chuyển hết về dạng 'patientXXX')
patient_ids = [10, 11, 12, 16, 17, 23, 28, 30, 33, 35, 38, 3, 40, 43, 5]

patients = [f"patient{pid:03d}" for pid in patient_ids]

# ======= hàm hỗ trợ =======
pat_set = set(patients)
fname_re = re.compile(r"^(patient\d{3})\.h5$", re.IGNORECASE)

def natural_key(name: str):
    """sắp xếp theo (patientID, frame số)"""
    m = fname_re.match(name)
    if not m:
        return (name, 10**9)  # đẩy các tên lạ xuống cuối
    pid = m.group(1)
    return (pid, 0)

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
