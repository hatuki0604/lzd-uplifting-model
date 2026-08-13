# Lazada Uplift Modeling — CATE Estimation Benchmark & Pipeline

Nghiên cứu thực nghiệm so sánh các phương pháp ước lượng **Conditional Average Treatment Effect (CATE)** với đặc tính Doubly Robust cho bài toán cá nhân hóa khuyến mãi (Uplift Modeling) trên dữ liệu Lazada. 

Dự án so sánh **DR-Learner family** (Kennedy 2020), các biến thể **Double Machine Learning (LinearDML, NonParamDML, CausalForestDML)** và hai mô hình baseline (**S-Learner, T-Learner**).

---

## 🏆 Kết quả thực nghiệm chính

- **Mô hình quán quân:** **DRLearner** (chọn dựa trên hệ số Qini đo trên tập `rct_select`).
- **Dữ liệu huấn luyện:** Gộp **926.669 dòng** từ `train` + `val`.
- **Đầu vào mô hình:** 76 đặc trưng (gồm 69 cột gốc sạch + 7 cột kỹ nghệ dẫn xuất).
- **Kết quả đánh giá RCT-holdout (mở đúng 1 lần):**
  - **Qini Score:** `0.0291` (Khoảng tin cậy 95% Bootstrap: `[0.0041, 0.0553]`).
  - **ATE trên RCT-holdout:** `0.00376` (tương đương `0.376` điểm phần trăm).
- **Ablation & Triển khai Production:**
  - **41 đặc trưng đầu vào sống** ($k=41$) đạt mốc an toàn tuyệt đối khi phục vụ (sai lệch dự đoán CATE = 0); 35 cột còn lại được điền giá trị mặc định.
  - Trong 41 cột này có 36 cột gốc dùng trực tiếp và 5 cột dẫn xuất do AI Service tính. Khi triển khai, service còn cần 4 cột nguồn `f7`, `f14`, `f17`, `f27` để tính đủ các cột dẫn xuất; vì vậy luồng nguồn cần tổng cộng **40 cột gốc**.

---

## ⚡ Hướng dẫn cài đặt & Chạy môi trường với `uv`

Dự án sử dụng **`uv`** (Astral) để quản lý phiên bản Python và khóa package dependency nhất quán giữa các máy.

### 1. Yêu cầu tiền đề
- [Git LFS](https://git-lfs.com/) (để tải dữ liệu lớn, file model và database MLflow).
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Công cụ quản lý Python siêu tốc).

Cài `uv` trên macOS/Linux nếu máy chưa có:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Tải repository & Tải file LFS
```bash
git clone https://github.com/hatuki0604/lzd-uplifting-model.git
cd lzd-uplifting-model

git lfs install
git lfs pull
```

### 3. Đồng bộ môi trường bằng `uv`
Repo tự động khóa phiên bản **Python 3.10.9** trong `.python-version`. Gõ lệnh sau để `uv` tự động tải Python 3.10 và tạo môi trường ảo `.venv`:

```bash
uv sync --frozen
```

*Lệnh trên sẽ tự động tạo thư mục `.venv` và cài chính xác các phiên bản thư viện đã khóa trong `uv.lock`. Không cần tự tạo virtual environment hay chạy `pip install` thủ công.*

### 4. Kiểm tra nhanh môi trường
```bash
uv run --frozen python -c "import cloudpickle, econml, lightgbm, mlflow, optuna, pandas; print('Môi trường OK!')"
```

---

## 📓 Chạy Notebooks & Khởi chạy Jupyter

Các notebook sử dụng đường dẫn tương đối từ thư mục `notebooks/`. Bạn khởi chạy Jupyter Lab bằng lệnh:

```bash
cd notebooks
uv run --project .. --frozen jupyter lab
```

### Chọn Kernel trong VS Code / Jupyter:
- **VS Code:** Chọn **Select Kernel $\rightarrow$ Python Environments $\rightarrow$ chọn `.venv/bin/python`** (macOS/Linux) hoặc `.venv\Scripts\python.exe` (Windows).
- **Working Directory:** Đảm bảo thư mục làm việc là `notebooks/`.

---

## 🗺️ Quy trình Pipeline 4 Notebooks

| Notebook | Nội dung chính | Đầu ra chính (Artifacts) |
|---|---|---|
| **`01_eda.ipynb`** | EDA, kiểm tra bias/overlap, chia dữ liệu phân tầng cố định | `split_assignment.parquet`, `eda_columns_to_drop.json` |
| **`02_feature_engineering.ipynb`** | Làm sạch, bỏ cột trùng/hằng số, tạo 3 nhóm đặc trưng mới, xuất 4 tập Parquet | `train/val/rct_select/rct_holdout.parquet`, `feature_info.json` |
| **`03_model_tuning.ipynb`** | Sub-sampling 200k/100k, xử lý overlap (trim theo Crump) + winsorize, Optuna tuning 6 mô hình CATE theo **DR-AUUC**, bootstrap KTC, log MLflow | `best_params.json`, `mlflow.db` |
| **`04_benchmark_evaluation.ipynb`** | Full re-train 926k dòng, đấu trên `rct_select`, Ablation study, đánh giá `rct_holdout`, đóng gói bàn giao | `model.pkl`, `model_full.pkl`, `metadata.json`, `feature_contract.json` |

*Lưu ý: Notebook 03 ở chế độ `full` chạy mất khoảng 48 phút. Tất cả output và artifacts đã được chạy sẵn và lưu trong repo, không cần chạy lại toàn bộ pipeline chỉ để xem kết quả.*

`rct_holdout` là tập đánh giá cuối đã được mở đúng một lần. Không dùng kết quả trên tập này để tuning lại mô hình hoặc chọn đặc trưng.

### Thước đo tinh chỉnh ở notebook 03

Bản đầu dùng **DR-MSE** làm hàm mục tiêu Optuna. Nó hỏng: khai triển ra thì

```
E[(τ̂ − Ỹ)²] = E[(τ̂ − τ)²]  +  E[Var(Ỹ | X)]
                ↑ thứ cần đo      ↑ hằng số, giống nhau ở mọi mô hình
```

Số hạng thứ hai không phụ thuộc mô hình nhưng lớn hơn số hạng đầu vài trăm lần, nên nó nuốt mất tín hiệu — sáu mô hình cho DR-MSE nằm trong `0,3231 – 0,3238` với mốc dự đoán hằng số `0,3234` nằm lọt giữa.

Bản hiện tại sửa bốn chỗ, theo nhận xét của mentor:

| # | Thay đổi | Chi tiết |
|---|---|---|
| 1 | **Xử lý overlap** | Trim hẳn vùng ngoài chồng lấn thay vì chỉ cắt `ê`. Ngưỡng `α` **không gõ tay** mà tính theo quy tắc Crump et al. (2009), có hàng rào `[0,02 – 0,10]` chặn hai đầu |
| 2 | **Winsorize** `Ỹ` ở phân vị 0,5% / 99,5% | Tách bạch `dr_ov` (chưa winsorize — dùng báo cáo **mức** ATE) và `dr_val` (đã winsorize — dùng làm nhãn **xếp hạng**) |
| 3 | **Hàm mục tiêu → DR-AUUC** | Sắp theo `τ̂`, lấy trung bình tích lũy của `Ỹ`, tính diện tích trên đường chéo, chuẩn hóa về thang 0–1 như hệ số Qini. Hướng đổi thành `maximize` |
| 4 | **Bootstrap theo cặp** | Chỉ tuyên bố A hơn B khi KTC 95% của *hiệu* `Qini_DR(A) − Qini_DR(B)` không chứa 0 |

**Estimand đã đổi** kèm theo thay đổi 1: thước đo ở notebook 03 nhắm vào `E[τ(X) | α ≤ e(X) ≤ 1−α]` — ATE trên **vùng overlap** của Val, không phải trên toàn quần thể Val. Notebook 03 phát biểu lại điều này ở Bước 10 và ghi vào `xu_ly_overlap.estimand` trong `best_params.json`.

Notebook 04 đọc `xu_ly_overlap.propensity_alpha` từ file đó để dùng đúng ngưỡng cắt `ê` mà notebook 03 đã tinh chỉnh. Trên tập RCT việc phát voucher là ngẫu nhiên (`e ≡ 0,5`) nên không cần trim, và quyết định quán quân vẫn thuộc về hệ số Qini đo ở đó.

### Nạp model đã bàn giao

Chạy từ thư mục gốc của repo:

```bash
uv run --frozen python -c "import pickle; model = pickle.load(open('artifacts/model.pkl', 'rb')); print(type(model).__name__)"
```

Artifact dùng `cloudpickle` khi xuất để class `DRLearner` định nghĩa trong notebook vẫn nạp được ở một process hoặc máy khác; phía sử dụng vẫn có thể gọi `pickle.load` như bình thường.

---

## 📊 Xem lịch sử thí nghiệm MLflow

Toàn bộ 66 runs thử nghiệm siêu tham số của 6 mô hình đã được lưu vết trong database SQLite `mlflow.db`. 

Từ thư mục gốc của dự án, khởi chạy MLflow UI:

```bash
uv run --frozen mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Sau đó truy cập trình duyệt tại địa chỉ: **`http://127.0.0.1:5000`**

---

## 📂 Cấu trúc Repository

```text
├── artifacts/              # Model binary (model.pkl), Metadata, Feature Contract, Best Params
├── dataset/                # Tập dữ liệu thô (csv) và 4 tập parquet đã chia (train/val/rct_select/rct_holdout)
├── notebooks/              # 4 Notebooks theo quy trình Data Science
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_tuning.ipynb
│   └── 04_benchmark_evaluation.ipynb
├── reports/figures/        # Biểu đồ kết quả xuất ra phục vụ báo cáo & slide
├── mlflow.db               # Database lưu lịch sử thí nghiệm MLflow
├── project-information.md  # Sơ đồ phân chia luồng công việc DS và DE
├── pyproject.toml          # Khai báo dependency dự án
├── uv.lock                 # File khóa chính xác phiên bản package
└── .python-version         # Khóa phiên bản Python 3.10.9
```

---

## 🔄 Quy tắc cập nhật Dependency

Không cài package trực tiếp bằng `pip` trong notebook. Khi cần thêm dependency mới:

```bash
uv add <ten-package>
```
Sau đó commit cả 2 file `pyproject.toml` và `uv.lock` lên Git.
