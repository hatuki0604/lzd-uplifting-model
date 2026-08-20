# Lazada Uplift Modeling — CATE Estimation Benchmark & Pipeline

Nghiên cứu thực nghiệm so sánh các phương pháp ước lượng **Conditional Average Treatment Effect (CATE)** với đặc tính Doubly Robust cho bài toán cá nhân hóa khuyến mãi (Uplift Modeling) trên dữ liệu Lazada.

Dự án so sánh **DR-Learner** (Kennedy 2020), ba biến thể **Double Machine Learning** (`LinearDML`, `NonParamDML`, `CausalForestDML`) và hai baseline (**S-Learner**, **T-Learner**).

---

## 🎯 Đích đến hiện tại

Bản đang bàn giao là **DR-Learner trên 30 đặc trưng**, huấn luyện trên toàn bộ 926.669 dòng `train + val`.

Đường đi tới nó có hai chặng:

1. **Chặng nghiên cứu** (`01` → `04c`): benchmark 6 phương pháp CATE, chọn quán quân trên `rct_select`, mở `rct_holdout` **đúng một lần** để đo. Kết quả: DR-Learner, 69 cột.
2. **Chặng rút gọn** (`05` → `09`): hỏi mô hình cần tối thiểu bao nhiêu cột, dựng lại thứ hạng đặc trưng trên tập Val để `rct_select` không vừa chọn vừa chấm, tinh chỉnh riêng cho tập cột rút gọn, rồi chấm qua một cổng quyết định khóa bằng mã băm. Kết quả: 30 cột.

Cuối cùng `10` suy ngưỡng phát voucher từ chính mô hình đang bàn giao.

**Cái bản 30 cột đánh đổi, nói thẳng:** nó **chưa từng được đo trên `rct_holdout`** và sẽ không được đo. `rct_holdout` đã mở đúng một lần cho bản 69 cột; mở lần hai là phá quy tắc của chính dự án. Bằng chứng ngoài mẫu duy nhất của bản đang chạy là `rct_select`, và giới hạn đó được ghi vào `artifacts/metadata.json → gioi_han` chứ không để trống. Bản 69 cột được giữ nguyên ở `artifacts/*_k69.*` để lùi lại bất cứ lúc nào.

> **README này cố tình không chép cứng con số nào.** Mỗi lần chạy lại, số nằm trong artifact và trong output notebook — chép sang đây là tạo thêm một chỗ để lệch. Đọc trường `run_mode` trước khi trích bất kỳ con số nào: `"smoke_test"` là chạy nhanh kiểm pipeline, **không dùng để báo cáo**; `"full"` mới là số báo cáo được.

---

## 🔢 Vì sao 83 đặc trưng còn 69

Notebook 01 tính ra danh sách này **từ dữ liệu**, rồi chốt bằng `assert` để lần chạy sau không lệch đi:

| Nhóm | Số cột | Cột | Tiêu chí |
|---|---|---|---|
| Hằng số | 1 | `f70` | chỉ một giá trị duy nhất trên toàn tập train |
| Trùng khít | 8 | `f15`, `f32`, `f33`, `f39`, `f71`, `f72`, `f77`, `f78` | giống hệt một cột khác ở **mọi** dòng (dò bằng băm nội dung cột) |
| Nhị phân dư thừa | 5 | `f67`, `f69`, `f73`, `f74`, `f76` | giống **hoặc bù** một cột khác ở ≥ 99,999% số dòng |

`83 − 1 − 8 − 5 = 69`. Mỗi nhóm luôn giữ lại một đại diện nên không mất tín hiệu nào — ví dụ nhóm `['f68','f69','f73','f74']` giữ `f68`.

Nhóm thứ ba cần dung sai `1e-5` chứ không dò được bằng đẳng thức tuyệt đối: `f69` và `f73` lệch nhau đúng vài dòng trên 741 nghìn, đủ để phép băm và phép so tương quan `±1` đều bỏ sót, trong khi về mặt tín hiệu chúng là một.

Danh sách này đi thẳng vào `artifacts/eda_columns_to_drop.json` và được đối chiếu lại ở các notebook sau cùng bộ test. **69 là không gian đặc trưng của dự án** — bản bàn giao 30 cột là một tập con của nó, và `metadata.json` giữ cả hai con số (`so_dac_trung` và `so_dac_trung_day_du`).

---

## ⚡ Cài đặt môi trường với `uv`

### 1. Yêu cầu tiền đề
- [Git LFS](https://git-lfs.com/) — để tải file dữ liệu lớn.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS/Linux
```

### 2. Tải repository & file LFS
```bash
git clone https://github.com/hatuki0604/lzd-uplifting-model.git
cd lzd-uplifting-model

git lfs install
git lfs pull
```

### 3. Đồng bộ môi trường
`.python-version` khóa **Python 3.10.9**. Lệnh sau tự tải Python 3.10 và tạo `.venv`:

```bash
uv sync --frozen
```

### 4. Kiểm tra nhanh
```bash
uv run --frozen python -c "import cloudpickle, econml, lightgbm, mlflow, optuna, pandas; print('Môi trường OK!')"
```

---

## 🗺️ Pipeline 11 notebook

Thứ tự chạy đúng bằng thứ tự đánh số:

```
01 → 02 → 03 → 04a → 04b → 04c → 05 → 07 → 08 → 09 → 10
└──────── chặng nghiên cứu ────────┘ └─ chặng rút gọn ─┘ └ ngưỡng
```

| Notebook | Nội dung chính | Đầu ra |
|---|---|---|
| **`01_eda.ipynb`** | **Chia dữ liệu trước khi nhìn vào nó** (seed 42, stratify theo `(is_treat, label)`), EDA, đo thiên vị chọn lọc, chốt danh sách 14 cột loại bỏ → 69 đặc trưng | `split_assignment.parquet`, `eda_columns_to_drop.json` |
| **`02_feature_engineering.ipynb`** | Áp danh sách cột loại bỏ, kiểm lại từng nhóm trên dữ liệu, kiểm tra positivity, cắt 4 tập theo đúng phân chia đã chốt. **Không tạo đặc trưng dẫn xuất** | `train/val/rct_select/rct_holdout.parquet`, `feature_info.json` |
| **`03_model_tuning.ipynb`** | Xử lý overlap (trim theo Crump) + winsorize, Optuna tinh chỉnh 6 mô hình theo **DR-AUUC** trên Val, bootstrap KTC theo cặp, ghi vết MLflow | `best_params.json`, `mlflow.db` |
| **`04a_train_models.ipynb`** | Huấn luyện 6 mô hình trên Train + Val, chấm CATE trên `rct_select`. Không xếp hạng, không nạp holdout | `04a_model_*.pkl`, `04a_cate_select.npz`, `04a_train_meta.json` |
| **`04b_select_model.ipynb`** | Chọn quán quân trên `rct_select`, gain + permutation importance, quét k **có huấn luyện lại từng mức**, rồi **khóa quyết định**. Không nạp holdout | `decisions_locked.json`, `04b_model_ban_giao.pkl`, `04b_select_meta.json` |
| **`04c_holdout_report.ipynb`** | Kiểm chuỗi 04a→04b→04c, mở `rct_holdout` **đúng một lần**, đóng gói bản 69 cột | `model.pkl`, `model_booster.txt`, `feature_contract.json`, `golden_predictions.csv`, `metadata.json` |
| **`05_quet_so_dac_trung.ipynb`** | Mô hình cần tối thiểu bao nhiêu cột? Quét k nhiều seed trên `rct_select`, đo **sàn nhiễu** của chính phép đo | `quet_dac_trung.json` |
| **`07_xep_hang_dac_trung_val.ipynb`** | Dựng lại thứ hạng đặc trưng **trên Val**, đối chiếu với thứ hạng của 04b, rồi tinh chỉnh riêng DR-Learner cho tập cột rút gọn. Không nạp holdout, không nạp `rct_select` | `xep_hang_val.json`, `best_params_k30.json` |
| **`08_so_sanh_k30.ipynb`** | So 4 nhánh nhiều seed trên `rct_select` + bootstrap theo cặp, chấm qua **cổng quyết định khóa bằng mã băm trước khi có số**. Không nạp holdout | `so_sanh_k30.json` |
| **`09_ban_giao_k30.ipynb`** | Huấn luyện bản rút gọn trên toàn bộ Train + Val, đóng gói artifact bàn giao. Dừng ngay nếu cổng của 08 không đạt | `model_k30.pkl`, `model_booster_k30.txt`, `feature_contract_k30.json`, `golden_predictions_k30.csv`, `metadata_k30.json` |
| **`10_chon_nguong.ipynb`** | Suy ngưỡng phát voucher từ đường cong Qini của **mô hình đang bàn giao**, kèm ràng buộc kinh tế | `nguong_quyet_dinh.json` |

### Vì sao `10` đứng cuối chứ không phải `06`

Notebook 10 suy ngưỡng từ phân vị điểm CATE, mà thang điểm là của **riêng từng mô hình**. Chạy nó trước `09` thì được ngưỡng của một mô hình sắp bị thay — file vẫn đọc được, con số vẫn hợp lệ, chỉ là nó thuộc về một phân bố điểm không còn tồn tại. Đây là lỗi im lặng, nên ngoài việc đánh số cho đúng thứ tự còn có `test_nguong_quyet_dinh_khong_bi_cu` chặn: ngưỡng phải khai đúng số cột và đúng mã băm cổng của bản đang bàn giao.

### Vì sao notebook 04 tách làm ba

Mỗi kernel Kaggle chỉ có 12 giờ, mà riêng `CausalForestDML` học trên toàn bộ 926 nghìn dòng đã ăn gần hết ngần ấy. Tách ra thì mỗi mảnh có quota riêng, và hỏng chỗ nào chỉ phải chạy lại chỗ đó.

Nhưng lý do đáng giá hơn nằm ở chỗ khác: **`rct_holdout.parquet` chỉ được nạp trong `04c`.** Không notebook nào khác trong repo mở file đó — kể cả `05`, `07`, `08`, `09`, `10`. "Lỡ nhìn holdout sớm" thành bất khả thi về mặt vật lý chứ không còn là một lời hứa trong markdown.

### Cỡ mẫu phải bằng nhau giữa 6 mô hình

`N_FIT` trong `04a` đặt số dòng tối đa mỗi mô hình được học. Nếu để khác nhau thì cột Qini trong bảng benchmark **trộn chất lượng phương pháp với lượng dữ liệu**, và câu hỏi nghiên cứu mất nghĩa. `04a_train_meta.json` ghi cờ `cung_co_mau`, và 04b cảnh báo nếu cờ đó là `false`.

---

## 🛡️ Bốn nguyên tắc chống rò rỉ, và chỗ thực thi từng cái

1. **Chia dữ liệu trước EDA.** Notebook 01 chia ngay ở ô lệnh thứ hai, trước cả `head()`, rồi xóa `val` và `rct_holdout` khỏi bộ nhớ. Phân chia ghi ra file kèm **mã băm** `data_id` từng tập; notebook 02 và bộ test băm lại rồi so.

2. **`rct_holdout` mở đúng một lần.** Mọi quyết định của chặng nghiên cứu nằm trong `04b` và được đóng băng vào `decisions_locked.json` kèm mã băm. `04c` là notebook duy nhất nạp file holdout; nó tính lại mã băm hai lần rồi dừng nếu khác. **Chặng rút gọn không mở lại file đó** — đó là lý do bản 30 cột không có số holdout.

3. **Ablation phải huấn luyện lại.** Mỗi mức k trong vòng quét là một mô hình huấn luyện lại trên đúng k cột đó. Cách cũ — giữ mô hình đầy đủ rồi điền trung vị — chỉ đo độ nhạy với giá trị giả, và với những cột mô hình chưa từng chia nhánh thì sai lệch bằng 0 là chuyện hiển nhiên, không phải bằng chứng.

4. **Tập cột không được vừa chọn vừa chấm.** Thứ hạng đặc trưng của `04b` đo trên `rct_select`; đem tập cột đó ra chấm lại trên chính `rct_select` là lập luận vòng tròn và thiên vị về phía bản rút gọn. Notebook `07` vì thế dựng lại thứ hạng **trên Val** — tập chưa từng dùng để quyết định gì về đặc trưng — để `rct_select` trở lại là tập giữ ngoài thật ở notebook `08`. Notebook `08` giữ lại nhánh dùng thứ hạng cũ để **đo** đúng độ lớn của thiên vị đó thay vì chỉ nói suông.

### Cổng quyết định của chặng rút gọn

Notebook `08` khóa ba điều kiện vào một dict rồi băm **trước khi tính bất kỳ con số nào**, và băm lại ở cuối — sửa cổng sau khi nhìn thấy kết quả thì notebook dừng. Notebook `09` băm lần nữa và từ chối đóng gói nếu cổng không đạt.

| Điều kiện | Kiểm gì |
|---|---|
| 1 | Hai thứ hạng độc lập (Val và `rct_select`) phải đồng ý về tập k cột ở mức tối thiểu định trước |
| 2 | Qini của nhánh ứng viên không tụt quá **sàn nhiễu** so với nhánh mốc |
| 3 | Bootstrap theo cặp trên `rct_select` không **chứng minh được** nhánh ứng viên tệ hơn |

**Hai thang nhiễu, đừng trộn vào nhau.** Sàn nhiễu đo dao động khi chỉ **đổi seed**; KTC bootstrap đo dao động khi **lấy mẫu lại dòng** của `rct_select`, và rộng hơn cả bậc độ lớn. Vì thế điều kiện 2 và 3 dùng hai mốc khác nhau. Đòi cận dưới bootstrap vượt `−sàn nhiễu` là đòi một phép kiểm mà `rct_select` không đủ công suất để qua — kể cả khi bản rút gọn thật sự tốt hơn.

Kèm theo đó, `so_sanh_k30.json` ghi **biên độ "không kém hơn" chặt nhất chứng nhận được**. Trên bộ dữ liệu này nó xấp xỉ bằng chính giá trị Qini, nghĩa là điều kiện 3 gần như không mang thông tin và quyết định thực chất dựa vào điều kiện 1 và 2. Con số đó nằm trong artifact để báo cáo nói thật về giới hạn của mình, thay vì để cổng trông chặt hơn thực tế.

**Phát biểu tối đa được phép rút ra là "không tệ hơn"** — không bao giờ là "tốt hơn". Chênh lệch nằm dưới sàn nhiễu thì không quy được cho việc bớt cột.

---

## 🔁 Chạy lại toàn bộ pipeline từ đầu

Đầu vào duy nhất là hai file trong `dataset/`:

```
dataset/full_trainset.csv     926.669 dòng — dữ liệu quan sát
dataset/full_testset.csv      181.669 dòng — dữ liệu RCT
```

Mọi thứ khác (4 file parquet, artifacts, biểu đồ, `mlflow.db`) là **sinh ra được** — xóa đi rồi chạy lại theo đúng thứ tự là có lại.

### Bước 0 — dọn output cũ (tùy chọn)

```bash
rm -rf artifacts/* reports/figures/*.png mlflow.db \
       dataset/train.parquet dataset/val.parquet \
       dataset/rct_select.parquet dataset/rct_holdout.parquet
```

Hai file CSV gốc **không được xóa**.

### Bước 1 — chạy nhanh để kiểm tra pipeline (khuyên dùng trước)

Mọi notebook nặng mặc định chạy ở chế độ nhanh. Không phải sửa gì trong notebook:

```bash
cd notebooks
for nb in 01_eda 02_feature_engineering 03_model_tuning \
          04a_train_models 04b_select_model 04c_holdout_report \
          05_quet_so_dac_trung 07_xep_hang_dac_trung_val \
          08_so_sanh_k30 09_ban_giao_k30 10_chon_nguong; do
  SMOKE_TEST=1 uv run --project .. --frozen \
    python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=5400 "$nb.ipynb"
done
```

Notebook 01 và 02 luôn chạy trên toàn bộ dữ liệu — chúng chỉ làm sạch và chia dữ liệu nên không có gì để rút gọn. Biến `SMOKE_TEST` không ảnh hưởng tới hai notebook này.

Muốn chạy bằng giao diện thì mở Jupyter rồi chạy lần lượt theo đúng thứ tự:

```bash
cd notebooks
uv run --project .. --frozen jupyter lab
```

### Bước 2 — chạy đầy đủ để lấy số báo cáo

Bật chế độ đầy đủ bằng **biến môi trường**, không sửa notebook:

```bash
cd notebooks
for nb in 03_model_tuning 04a_train_models 04b_select_model 04c_holdout_report \
          05_quet_so_dac_trung 07_xep_hang_dac_trung_val \
          08_so_sanh_k30 09_ban_giao_k30 10_chon_nguong; do
  SMOKE_TEST=0 uv run --project .. --frozen \
    python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=36000 "$nb.ipynb"
done
```

| Notebook | `SMOKE_TEST=1` (mặc định) | `SMOKE_TEST=0` |
|---|---|---|
| `03` | 11 trial, mẫu 20k/10k | **66 trial** (12 mỗi mô hình, 6 cho CausalForest), mẫu 200k/100k |
| `04a` | cả 6 mô hình trên 20k dòng | cả 6 mô hình trên **toàn bộ** Train + Val |
| `04b` | bootstrap 100, quét 4 mức k trên 20k dòng | bootstrap 1.000, quét 5 mức k trên 200k dòng |
| `04c` | bootstrap 100 | bootstrap 1.000 |
| `05` | 3 mức k, 2 seed, 50k dòng | 14 mức k, 5 seed, 200k dòng |
| `07` | 2 trial, 2 lần xáo mỗi cột, mẫu 20k/10k | 12 trial, 5 lần xáo mỗi cột, mẫu 200k/100k |
| `08` | 4 nhánh × 2 seed, 50k dòng, bootstrap 100 | 4 nhánh × 5 seed, 200k dòng, bootstrap 1.000 |
| `09` | huấn luyện trên 50k dòng | huấn luyện trên **toàn bộ** Train + Val |
| `10` | bootstrap 60 | bootstrap 400 |

Vòng quét k ở `04b` và `05` có trần cỡ mẫu **riêng**, không dùng lại cỡ mẫu của `04a`: vòng quét chỉ so *tương đối* giữa các mức k nên vẫn có nghĩa ở cỡ mẫu nhỏ hơn. Đổi lại, mốc so sánh của vòng quét là mốc `k` đầy đủ của **chính vòng quét đó** — để chênh lệch chỉ đến từ số đặc trưng chứ không lẫn cỡ mẫu.

Chế độ đầy đủ tốn hàng giờ. Con số thời gian phụ thuộc máy nên không ghi ở đây; ô lệnh cuối mỗi notebook in ra thời gian thực tế.

### Bước 3 — chạy bộ test

```bash
uv run --frozen python -m unittest tests.test_pipeline_contract -v
```

---

## ☁️ Chạy trên Kaggle

Notebook 01 và 02 **chạy ở máy local** — chúng chỉ đọc hai file CSV gốc và mất vài phút, mà phép chia dữ liệu là thứ duy nhất không sửa lại được nếu sai. Các mảnh nặng thì đẩy lên Kaggle.

```bash
./kaggle_run.sh data          # đẩy 4 parquet + feature_info.json thành Kaggle dataset (~61 MB)

./kaggle_run.sh push 03       # tinh chỉnh siêu tham số
./kaggle_run.sh status 03     # đợi tới khi complete
./kaggle_run.sh push 04a      # huấn luyện 6 mô hình        (gắn 03)
./kaggle_run.sh push 04b      # chọn quán quân + đặc trưng   (gắn 03, 04a)
./kaggle_run.sh push 04c      # holdout + đóng gói 69 cột    (gắn 04a, 04b)
./kaggle_run.sh push 05       # quét số đặc trưng            (gắn 03, 04b)
./kaggle_run.sh push 07       # xếp hạng trên Val + tinh chỉnh (gắn 03, 04b, 05)
./kaggle_run.sh push 08       # so 4 nhánh + cổng quyết định  (gắn 03, 05, 07)
./kaggle_run.sh push 09       # đóng gói bản rút gọn          (gắn 03, 04b, 07, 08)

./kaggle_run.sh pull 09       # tải artifact bàn giao về kaggle_output/09/
```

Notebook `10` chạy local sau cùng, vì nó cần bộ artifact bàn giao đã nằm đúng chỗ trong `artifacts/`.

Bốn điều cần nhớ:

- **Phải đợi kernel trước `complete` rồi mới đẩy kernel sau**, vì Kaggle gắn output của kernel trước qua `kernel_sources`.
- Script tự chèn một ô lệnh đặt `SMOKE_TEST=0` vào **đầu** notebook khi đẩy lên — đã đẩy lên là để chạy thật. Muốn thử nhanh: `KAGGLE_SMOKE=1 ./kaggle_run.sh push 08`. Vì ô lệnh này được *chèn thêm* chứ không thay thế, **notebook trong repo không được có sẵn ô đặt `SMOKE_TEST`** — nó sẽ chạy sau và ghi đè ý định của script.
- `pull` tải về `kaggle_output/<nb>/`, **không** chép thẳng vào `artifacts/`. Chép tay sau khi đã xem qua.
- Kaggle chạy Python 3.12 và tạo một `mlflow.db` **mới** trong `/kaggle/working`. File tải về chỉ chứa experiment của lần chạy đó — **đừng chép đè lên `mlflow.db` ở gốc repo**, sẽ mất 66 run của notebook 03.

---

## 📦 Bàn giao cho DE

Notebook `09` xuất bộ artifact `*_k30`, sau đó bộ này được đề bạt thành tên chuẩn mà DE tiêu thụ. Tất cả nói về **cùng một mô hình, cùng một danh sách cột, cùng một thứ tự**:

| File | Nội dung |
|---|---|
| `artifacts/model.pkl` | Mô hình bàn giao, huấn luyện trên đúng tập đặc trưng đã chốt |
| `artifacts/model_booster.txt` | Cùng mô hình đó ở định dạng text của LightGBM |
| `artifacts/feature_contract.json` | Danh sách cột đầu vào kèm `thu_tu_dua_vao_mo_hinh`, giá trị mặc định, khoảng giá trị, hạng quan trọng |
| `artifacts/golden_predictions.csv` | 1.000 dòng `data_id` + `cate` từ `rct_select` để DE tự đối chiếu |
| `artifacts/metadata.json` | Siêu tham số, cổng quyết định, bằng chứng trên `rct_select`, **trường `gioi_han`**, thông tin đóng gói |
| `artifacts/nguong_quyet_dinh.json` | Ngưỡng phát voucher, suy từ chính mô hình trên (notebook 10) |
| `artifacts/*_k69.*` | Bản 69 cột của chặng nghiên cứu — giữ để lùi lại, cách lùi ghi trong `metadata.json → lich_su_ban_giao` |

Contract **không chứa cột `fe_*`** nào — pipeline này không tạo đặc trưng dẫn xuất, nên AI Service không phải tính lại công thức nào lúc phục vụ. Số cột trong contract, số cột `model.pkl` nhận vào và số cột của booster phải khớp nhau; cả ba đều có `assert` trong notebook và trong bộ test.

### ⚠️ Đọc `metadata.json → gioi_han` trước khi trích số

Bản đang bàn giao đi qua chặng rút gọn nên **không có metric trên `rct_holdout`** — và `metadata.json` cố tình không mang trường `metric_rct_holdout` để không ai trích nhầm. Bộ test chặn luôn việc thêm trường đó vào. Số holdout của bản 69 cột nằm ở `artifacts/metadata_k69.json`, và nó thuộc về **mô hình khác**.

### Nạp model

Đường **chắc chắn chạy được ở mọi nơi** là booster dạng text:

```python
import lightgbm as lgb
booster = lgb.Booster(model_file='artifacts/model_booster.txt')
```

> ⚠️ **`model.pkl` phụ thuộc phiên bản Python.** `cloudpickle` đóng gói class kèm cả code object, mà cấu trúc code object **không tương thích giữa các bản Python**. Phiên bản đóng gói ghi ở `metadata.json → serialization.python_dong_goi`, và **phải khớp với Python đang chạy** — lệch bản sẽ báo `TypeError: code expected at most 16 arguments`.
>
> Artifact hiện tại được đóng gói trên Kaggle (**Python 3.12**), trong khi `.python-version` của repo khóa **3.10.9**. Nghĩa là `uv run --frozen python -c "import pickle; pickle.load(...)"` sẽ **không** nạp được `model.pkl` — đây là hệ quả của việc chạy notebook trên Kaggle, không phải lỗi cấu hình. Dùng `model_booster.txt`, hoặc nạp `model.pkl` bằng đúng bản Python ghi trong metadata.

Cách ghép CATE phụ thuộc cấu trúc mô hình, và được ghi rõ trong `metadata.json → booster.cach_ghep_cate`:

| Mô hình | File | Số cột đầu vào | Ghép CATE |
|---|---|---|---|
| DR-Learner | `model_booster.txt` | = số cột contract | `booster.predict(X)` |
| S-Learner | `model_booster.txt` | = số cột contract **+ 1** | `p(X,1) − p(X,0)`, cột cuối là `is_treat` |
| T-Learner | `model_booster.txt` + `model_booster_control.txt` | = số cột contract | `p1(X) − p0(X)` |
| LinearDML / NonParamDML / CausalForestDML | *không có* | — | không có dạng text của LightGBM; chỉ bàn giao được `model.pkl` |

`X` phải dựng theo đúng `thu_tu_dua_vao_mo_hinh` trong `feature_contract.json`. Sai thứ tự cột cho dự đoán sai mà **không báo lỗi** — đó là lý do có `golden_predictions.csv`.

### Kiểm tra phía DE bằng golden predictions

```python
import json
import numpy as np, pandas as pd, lightgbm as lgb

ct = json.load(open('artifacts/feature_contract.json', encoding='utf-8'))
cols = [d['ten'] for d in sorted(ct['dac_trung'], key=lambda d: d['thu_tu_dua_vao_mo_hinh'])]

golden = pd.read_csv('artifacts/golden_predictions.csv', float_precision='round_trip')
rct = pd.read_parquet('dataset/rct_select.parquet').set_index('data_id')

X = rct.loc[golden['data_id'], cols].to_numpy(dtype=np.float64)
booster = lgb.Booster(model_file='artifacts/model_booster.txt')

assert np.array_equal(booster.predict(X), golden['cate'].values)   # phải bằng 0 tuyệt đối
print('OK —', len(golden), 'dự đoán khớp từng bit')
```

**`float_precision='round_trip'` là bắt buộc.** Bộ phân tích CSV mặc định của pandas nhanh hơn nhưng lệch ở chữ số cuối, đủ để phép so bằng tuyệt đối ở trên fail — kiểm chứng được trên chính file này. Ngược lại, dtype của `X` **không** quan trọng: các cột đặc trưng trong parquet vốn là `float32`, nên ép sang `float32` hay `float64` đều cho cùng giá trị.

---

## 🧪 Bộ test

```bash
uv run --frozen python -m unittest tests.test_pipeline_contract -v
```

`tests/test_pipeline_contract.py` không huấn luyện lại gì, chỉ kiểm những bất biến đã từng lệch âm thầm:

| Nhóm | Kiểm gì |
|---|---|
| Số đặc trưng | Đúng 69 ở `eda_columns_to_drop.json`, `feature_info.json`, `best_params.json` và 4 file parquet; đúng danh sách 14 cột loại bỏ |
| Không có `fe_*` | Không artifact nào — kể cả `feature_contract.json` — chứa cột dẫn xuất |
| Phép chia không đổi | Chia lại từ hai file CSV bằng seed 42 phải ra đúng mã băm `data_id` của `split_assignment.parquet` |
| Thứ tự đặc trưng | Thứ tự cột parquet, `thu_tu_dua_vao_mo_hinh` liên tục từ 0, contract khớp metadata, contract theo thứ tự cột gốc chứ không theo độ quan trọng |
| Prediction parity | `model.pkl` nạp lại và `model_booster.txt` đều tái lập đúng từng bit `golden_predictions.csv`; số cột booster khớp contract |
| Chuỗi `04a → 04b → 04c` | `run_id` nối đúng, mã băm đặc trưng khớp cả ba mảnh — **chỉ chạy khi bản bàn giao đến từ chặng nghiên cứu** |
| Chuỗi `07 → 08 → 09` | Ba file bằng chứng phải là bản `full`, cổng phải đạt, **mã băm cổng phải tính lại đúng từ chính định nghĩa cổng**, tập cột đang giao phải đúng tập của nhánh ứng viên đã chấm, siêu tham số phải đúng bộ đã tinh chỉnh cho tập cột đó, và metadata phải ghi rõ `gioi_han` đồng thời **không** mang `metric_rct_holdout` |
| Ngưỡng không bị cũ | `nguong_quyet_dinh.json` phải khai đúng số cột và đúng mã băm cổng của mô hình đang bàn giao |
| File khóa 04b | `decisions_locked.json` luôn phải tự nhất quán — mã băm tính lại được từ chính nội dung đã khóa |
| 66 trial | Khi `run_mode = "full"`, tổng số trial của notebook 03 phải đúng 66 |

Test nào thiếu artifact đầu vào thì tự bỏ qua kèm lời nhắc chạy notebook nào — nên chạy được ngay cả khi mới xong notebook 01. Các test dành riêng cho một chặng sẽ tự bỏ qua khi bản bàn giao đến từ chặng kia.

---

## 📊 Xem lịch sử thí nghiệm MLflow

```bash
uv run --frozen mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Sau đó mở **`http://127.0.0.1:5000`**.

| Experiment | Nguồn | Số run ở chế độ đầy đủ |
|---|---|---|
| `lazada_cate_tuning` | notebook 03 — 6 mô hình | 66 |
| `lazada_cate_tuning_k30` | notebook 07 — riêng DR-Learner trên tập cột rút gọn | 12 |

Mỗi run mang tag `run_mode` để không lẫn chạy nhanh với chạy đầy đủ. Khi notebook 07 chạy trên Kaggle, experiment của nó nằm trong `mlflow.db` **của kernel đó** (`kaggle_output/07/mlflow.db`) chứ không tự gộp vào file ở gốc repo.

---

## 📂 Cấu trúc Repository

```text
├── artifacts/              # Toàn bộ file bàn giao — sinh ra bởi notebook, không commit tay
│   ├── split_assignment.parquet    # phép chia 4 tập (notebook 01)
│   ├── eda_columns_to_drop.json    # 69 đặc trưng sạch + 14 cột loại bỏ (notebook 01)
│   ├── feature_info.json           # hợp đồng đặc trưng (notebook 02)
│   ├── best_params.json            # tinh chỉnh 6 mô hình (notebook 03)
│   ├── decisions_locked.json       # quyết định khóa trước khi mở holdout (notebook 04b)
│   ├── quet_dac_trung.json         # vòng quét số đặc trưng (notebook 05)
│   ├── xep_hang_val.json           # thứ hạng đặc trưng đo trên Val (notebook 07)
│   ├── best_params_k30.json        # tinh chỉnh riêng cho tập cột rút gọn (notebook 07)
│   ├── so_sanh_k30.json            # 4 nhánh + cổng quyết định (notebook 08)
│   ├── nguong_quyet_dinh.json      # ngưỡng phát voucher (notebook 10)
│   ├── model.pkl / model_booster.txt / feature_contract.json
│   ├── golden_predictions.csv / metadata.json
│   ├── *_k30.*                     # bản gốc do notebook 09 xuất ra
│   └── *_k69.*                     # bản 69 cột, giữ để lùi lại
├── dataset/                # 2 file CSV gốc + 4 tập parquet sinh ra từ notebook 02
├── notebooks/              # 11 notebook, thứ tự chạy = thứ tự đánh số
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_tuning.ipynb
│   ├── 04a_train_models.ipynb
│   ├── 04b_select_model.ipynb
│   ├── 04c_holdout_report.ipynb
│   ├── 05_quet_so_dac_trung.ipynb
│   ├── 07_xep_hang_dac_trung_val.ipynb
│   ├── 08_so_sanh_k30.ipynb
│   ├── 09_ban_giao_k30.ipynb
│   └── 10_chon_nguong.ipynb
├── scripts/                # chon_nguong.py — bản script độc lập của notebook 10
├── demo_ba/                # demo cho BA + artifact_check.py đối chiếu bàn giao với repo DE
├── tests/                  # test_pipeline_contract.py — chốt cổng cho artifact
├── kaggle_run.sh           # đẩy notebook lên Kaggle chạy theo chuỗi kernel
├── reports/figures/        # Biểu đồ xuất ra phục vụ báo cáo & slide
├── mlflow.db               # Lịch sử thí nghiệm MLflow
├── project-information.md  # Sơ đồ phân chia luồng công việc DS và DE
├── pyproject.toml          # Khai báo dependency
├── uv.lock                 # Khóa chính xác phiên bản package
└── .python-version         # Khóa Python 3.10.9
```

---

## 🔄 Quy tắc cập nhật Dependency

Không cài package trực tiếp bằng `pip` trong notebook. Khi cần thêm dependency:

```bash
uv add <ten-package>
```

Sau đó commit cả `pyproject.toml` và `uv.lock`.
