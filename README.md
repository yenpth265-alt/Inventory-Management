# 📦 Inventory Optimization v2 — Data-Driven & RL-Based Multi-Item Inventory Management

> **Mục tiêu nghiên cứu:** Xây dựng khung tối ưu hóa tồn kho đa sản phẩm thế hệ mới, tách biệt hoàn toàn hai mô-đun **Dự báo (Forecasting)** và **Tối ưu hóa (Inventory Optimization)**, đồng thời khảo sát tính khả thi của **Học tăng cường (Reinforcement Learning)** trong quản trị tồn kho. 

---

## 📌 Mục lục

1. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
2. [Cài đặt môi trường](#3-cài-đặt-môi-trường)
3. [Quy tắc làm việc nhóm (Git Workflow)](#4-quy-tắc-làm-việc-nhóm-git-workflow)

---


## 1. Cấu trúc thư mục

```
Inventory_Optimization_v2/
│
├── docs/                          # Tài liệu nghiên cứu
│   ├── literature_review/         # Bảng phân tích bài báo tham khảo
│   ├── meeting_notes/             # Biên bản họp
│   ├── references/                # File PDF bài báo gốc
│   └── diagrams/                  # Sơ đồ kiến trúc hệ thống
│
├── research/                      # Phân tích lý thuyết & định hướng
│   ├── 01_e2e_optimization/       # Nghiên cứu về P&O, DFL, SPO
│   ├── 02_forecasting/            # Probabilistic Forecasting methods
│   ├── 03_inventory_optimization/ # Classical & Modern inventory models
│   ├── 04_reinforcement_learning/ # RL for inventory: DQN, PPO, SAC
│   └── 05_integration_study/      # Cách tích hợp Forecast + Optimization
│
├── data/
│   ├── raw/                       # Dữ liệu gốc chưa xử lý (KHÔNG commit lên git)
│   ├── processed/                 # Dữ liệu đã làm sạch (file .parquet)
│   └── external/                  # Bảng tham số kinh tế (Gross Margin, Lead time)
│
├── notebooks/
│   ├── exploration/               # EDA, khám phá dữ liệu ban đầu
│   ├── experiments/               # Thử nghiệm mô hình
│   └── baselines/                 # Chạy mô hình baseline để so sánh
│
├── src/                           # Source code chính thức (đã review)
│   ├── forecasting/               # Module A: Probabilistic Forecasting
│   ├── inventory/                 # Module B: Inventory Policy Simulation
│   ├── rl_agent/                  # Module C: RL Agent (PPO/SAC/DQN)
│   ├── optimization/              # Bài toán MILP / Knapsack
│   ├── evaluation/                # Đo lường: WRMSSE, Fill Rate, GMROI
│   └── utils/                     # Hàm dùng chung (I/O, logging, seed)
│
├── experiments/
│   ├── configs/                   # File cấu hình thực nghiệm (YAML)
│   ├── results/                   # Kết quả thực nghiệm (JSON, CSV)
│   └── logs/                      # Log quá trình train/run
│
├── tests/                         # Unit test cho các module trong src/
│
├── skills/                        # Hướng dẫn & cẩm nang tác nghiệp cho AI Agent
│
├── .github/
│   ├── ISSUE_TEMPLATE/            # Template tạo Issue
│   └── PULL_REQUEST_TEMPLATE/     # Template tạo Pull Request
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 2. Cài đặt môi trường

### Yêu cầu
- Python **>= 3.10**
- Khuyến nghị: **Anaconda** hoặc **Python venv**

### Các bước cài đặt

```bash
# 1. Clone repo về máy
git clone https://github.com/<org>/Inventory_Optimization_v2.git
cd Inventory_Optimization_v2

# 2. Tạo môi trường ảo
python -m venv venv

# 3. Kích hoạt môi trường ảo
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Cài đặt thư viện
pip install -r requirements.txt
```

### Lưu ý về dữ liệu
- **KHÔNG commit** file dữ liệu nặng lên Git (đã được chặn bởi `.gitignore`).
- Dữ liệu được chia sẻ qua **Google Drive của nhóm**. Sau khi tải về, đặt vào thư mục `data/raw/`.
- Hỏi **Technical Lead (Nam Hải)** để lấy link Google Drive.

---

## 3. Quy tắc làm việc nhóm (Git Workflow)

> ⚠️ **Đọc kỹ và tuân thủ nghiêm ngặt. Vi phạm quy tắc này có thể làm mất code của cả nhóm.**

### 4.1. Mô hình phân nhánh (Branching Strategy)

Dự án sử dụng mô hình **Feature Branch Workflow** với 3 tầng nhánh:

```
main          ← Nhánh chính thức. CHỈ chứa code ổn định, đã được review.
  └── dev     ← Nhánh tích hợp. Tất cả feature branch được merge vào đây trước.
        └── feature/xxx  ← Nhánh công việc cá nhân của từng thành viên.
```

**Quy tắc đặt tên nhánh:**
```
feature/<ten-nguoi>/<mo-ta-ngan>
fix/<ten-nguoi>/<mo-ta-loi>
docs/<ten-nguoi>/<mo-ta>

# Ví dụ:
feature/namhai/milp-dual-constraint
feature/haiyen/lgbm-quantile-forecast
docs/tonghang/literature-review-rl
fix/linhchi/ablation-study-metrics
```

### 4.2. Phân quyền rõ ràng

| Hành động | Nam Hải | Hải Yến / Tống Hằng / Linh Chi / Diễm Quỳnh |
| :--- | :---: | :---: |
| **Push trực tiếp vào `main`** | ✅ | ❌ NGHIÊM CẤM |
| **Push trực tiếp vào `dev`** | ✅ | ❌ NGHIÊM CẤM |
| **Push vào nhánh `feature/` của mình** | ✅ | ✅ |
| **Tạo Pull Request (PR) vào `dev`** | ✅ | ✅ |
| **Review và Approve PR** | ✅ | ❌ |
| **Merge PR vào `dev`** | ✅ | ❌ |
| **Merge `dev` vào `main`** | ✅ | ❌ |

### 4.3. Quy trình làm việc hàng ngày (Step-by-step)

**BƯỚC 1: Bắt đầu một nhiệm vụ mới**
```bash
# Luôn luôn kéo code mới nhất về trước khi làm việc
git checkout dev
git pull origin dev

# Tạo nhánh mới từ dev (KHÔNG bao giờ tạo từ main)
git checkout -b feature/namhai/milp-dual-constraint
```

**BƯỚC 2: Làm việc và commit thường xuyên**
```bash
# Thêm file vào staging (chỉ add những file liên quan)
git add src/optimization/milp_solver.py
git add notebooks/experiments/exp_milp_v1.ipynb

# Commit với message rõ ràng theo chuẩn Conventional Commits
git commit -m "feat(optimization): thêm ràng buộc thể tích kho V_max vào MILP"
```

**Chuẩn commit message:**
```
feat(scope): thêm tính năng mới
fix(scope): sửa lỗi
docs(scope): cập nhật tài liệu
refactor(scope): tái cấu trúc code không đổi logic
test(scope): thêm unit test
exp(scope): thực nghiệm (không vào code chính thức)

# Ví dụ thực tế:
feat(forecasting): implement LightGBM Quantile Regression q50/q95
fix(inventory): sửa công thức Safety Stock bị sai khi L > 0
docs(research): bổ sung phân tích bài báo SPO của Elmachtoub 2022
exp(rl): thử nghiệm PPO agent trên môi trường tồn kho đơn giản
```

**BƯỚC 3: Đẩy nhánh lên GitHub và tạo Pull Request**
```bash
# Đẩy nhánh lên GitHub
git push origin feature/namhai/milp-dual-constraint

# Sau đó vào GitHub → tạo Pull Request từ nhánh này vào dev
# Điền đầy đủ template PR (xem bên dưới)
# Tag @namhai để review
```

**BƯỚC 4: Xử lý xung đột (Conflict) khi Pull/Push**
```bash
# Tình huống: Bạn đang làm feature branch, dev đã có update mới
# Không được merge bừa — hãy làm theo đúng quy trình này:

# 1. Kéo code mới nhất từ dev về
git fetch origin
git checkout dev
git pull origin dev

# 2. Quay lại nhánh của mình và rebase lên dev mới
git checkout feature/ten-nhanh-cua-ban
git rebase origin/dev

# 3. Nếu có conflict, VS Code sẽ highlight — giải quyết từng file một
# Sau khi sửa xong:
git add <file-da-sua>
git rebase --continue

# 4. Đẩy lại nhánh (cần dùng --force-with-lease vì lịch sử đã thay đổi)
git push origin feature/ten-nhanh-cua-ban --force-with-lease
```

> ⚠️ **NGHIÊM CẤM dùng `git push --force` (không có `--lease`)**. Chỉ dùng `--force-with-lease` để tránh ghi đè công việc của người khác.

### 4.4. Quy tắc Pull Request (PR)

- **Mỗi PR chỉ giải quyết MỘT nhiệm vụ duy nhất** (không nhét nhiều việc vào 1 PR).
- **Phải điền đầy đủ PR Template** (mô tả, checklist, screenshot nếu có).
- **Tech Lead (Nam Hải) sẽ review trong vòng 24–48 giờ**. Nếu cần sửa, reviewer để lại comment cụ thể.
- **PR được merge khi:** Có ít nhất 1 Approve từ Tech Lead VÀ tất cả CI checks pass.

### 4.5. Các điều NGHIÊM CẤM

```
❌ NGHIÊM CẤM push thẳng vào nhánh main hoặc dev
❌ NGHIÊM CẤM commit file dữ liệu nặng (*.csv, *.zip, *.parquet)
❌ NGHIÊM CẤM commit thông tin bí mật (API key, password, credentials)
❌ NGHIÊM CẤM dùng git push --force (chỉ được dùng --force-with-lease)
❌ NGHIÊM CẤM merge PR của chính mình mà chưa được review
```
