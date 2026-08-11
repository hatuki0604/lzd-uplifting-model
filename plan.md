# Kế hoạch Triển khai — Nghiên cứu Doubly Robust CATE trên Lazada Uplift Dataset

Tài liệu này mô tả chi tiết những gì sẽ làm, làm theo thứ tự nào, và mỗi bước xong thì kiểm tra bằng gì.

---

## 0. Bối cảnh & Nguyên tắc

**Phạm vi:** chỉ phần Data Science. Toàn bộ hạ tầng (ERD, Airflow, Redis, MinIO, FastAPI, Docker) thuộc về DE, không nằm trong tài liệu này.

**Hình thức:** 4 notebook Jupyter, không có file `.py` nào. Các notebook nối nhau bằng file trên đĩa (parquet / json), không import lẫn nhau. Mỗi notebook chạy độc lập từ trên xuống dưới.

**Ba nguyên tắc xuyên suốt:**

1. **Xây thước đo trước khi xây mô hình.** Không có metric đáng tin thì tinh chỉnh siêu tham số là vô nghĩa.
2. **Tập RCT-holdout là bất khả xâm phạm.** Chỉ chạm vào đúng một lần, ở bước cuối cùng. Nếu lỡ nhìn kết quả rồi quay lại sửa mô hình thì con số đó mất giá trị khoa học.
3. **Mọi con số ngẫu nhiên đều có seed.** Chạy lại phải ra đúng kết quả cũ.

---

## 1. Vấn đề của phiên bản hiện tại

Cần nêu rõ để biết tại sao phải làm lại:

| Vấn đề | Hậu quả |
|---|---|
| Không có tập Validation | Đang chọn mô hình dựa trên chính tập test — như xem trước đề thi |
| Siêu tham số gõ tay, không tinh chỉnh | Không chứng minh được mô hình nào thực sự tốt hơn hay chỉ do may |
| MLflow chỉ có 4 run, không log siêu tham số nào | Mất ý nghĩa của việc quản lý thí nghiệm |
| Metric chưa được kiểm chứng | Nếu công thức Qini sai thì toàn bộ kết luận sai theo |
| Không lưu mô hình ra file | Không có gì để bàn giao, không tái lập được |
| Thiếu ATE và mô phỏng chính sách | Thiếu con số nghiệp vụ mà hội đồng dễ hiểu nhất |

---

## 2. Thiết kế chia dữ liệu

Đây là quyết định quan trọng nhất của cả dự án.

```
full_trainset.csv  (926.669 dòng — Observational, có selection bias)
        │
        ├── Train      80%  ~741k   →  huấn luyện mô hình
        └── Val        20%  ~185k   →  tinh chỉnh siêu tham số

full_testset.csv   (181.669 dòng — RCT, không bias)
        │
        ├── RCT-select   50%  ~91k  →  chọn quán quân giữa các mô hình
        └── RCT-holdout  50%  ~91k  →  chạy MỘT lần, ra số báo cáo cuối
```

**Vì sao chia như vậy:** mentor yêu cầu 80/10/10 với Val để tinh chỉnh và Test để các mô hình "đấu" nhau. Bộ dữ liệu này đã có sẵn hai file tách biệt về bản chất (một bên có bias, một bên RCT), nên áp dụng nguyên tắc đó theo cách phù hợp: Val cắt từ dữ liệu quan sát, còn tập RCT cắt đôi để tách khâu *chọn mô hình* khỏi khâu *báo cáo kết quả*.

**Chia stratify theo cặp `(is_treat, label)`** vì tỉ lệ chuyển đổi chỉ khoảng 2%. Chia ngẫu nhiên thuần dễ làm lệch tỉ lệ giữa các tập.

**Xử lý vấn đề tập Val có bias:** không thể tính Qini thô trên dữ liệu có selection bias, vì thứ hạng sẽ bị bóp méo bởi việc ai được phát voucher. Thay vào đó dùng **DR-score** — chính pseudo-outcome doubly robust — làm nhãn thay thế, rồi đo sai số bình phương giữa CATE dự đoán và DR-score. Đây là cách chuẩn trong tài liệu để đánh giá CATE khi không có RCT.

---

## 3. Notebook 01 — Exploratory Data Analysis

**Nạp vào:** `dataset/full_trainset.csv`, `dataset/full_testset.csv`

### Phong cách trình bày

Viết theo lối EDA từng bước quen thuộc trên Kaggle, thay vì gom thành vài ô lệnh lớn:

- Bắt đầu bằng việc **nhìn dữ liệu thật**: `head()`, `tail()`, `info()`, `describe()`, `value_counts()`, `nunique()`. Không phân tích gì cao siêu trước khi biết dữ liệu trông ra sao.
- **Mỗi ô lệnh trả lời đúng một câu hỏi**, ngắn gọn 3–10 dòng. Người đọc chạy tới đâu hiểu tới đó.
- **Markdown ngắn** — một tiêu đề bước và một hai câu diễn giải, không viết thành bài luận.
- Ưu tiên **in bảng kết quả thô** để tự nhìn ra vấn đề, thay vì chỉ đưa kết luận đã tóm tắt sẵn.
- Biểu đồ đơn giản, có nhãn số trên cột.

Bảng `describe().T` đầy đủ của toàn bộ đặc trưng cũng chính là thứ mentor yêu cầu đưa vào slide: phân phối và thang đo của các biến để chuẩn bị cho bước kỹ nghệ đặc trưng.

### Bảy phần, khoảng 40 bước nhỏ

**Phần 1 — Làm quen với dữ liệu**
Nạp file, xem `head` và `tail`, `shape`, `info`, kiểm tra `data_id` có trùng không, đếm giá trị khuyết thiếu, `describe().T` cho biến chính và cho toàn bộ đặc trưng, `nunique` để đoán kiểu biến, phân loại nhị phân / rời rạc / liên tục, tìm cột hằng số, tìm cột trùng lặp bằng băm nội dung, chốt danh sách cột loại bỏ.

**Phần 2 — Treatment và Outcome**
`value_counts` của `is_treat` và `label` trên cả hai tập, bảng chéo `W × Y`, tỉ lệ chuyển đổi theo nhóm, biểu đồ so sánh. Kết thúc bằng phép so sánh quan trọng nhất: **ước lượng ngây thơ trên dữ liệu quan sát so với hiệu ứng thật đo từ RCT**.

**Phần 3 — Thiên vị chọn lọc**
So sánh trực tiếp bảng trung bình của nhóm `W=1` và `W=0` trước, rồi mới chuẩn hóa thành SMD. Tính SMD cho **cả hai tập** — tập RCT đóng vai trò kiểm chứng công thức, vì nếu tính đúng thì SMD trên RCT phải gần 0. Love plot và biểu đồ phân phối của các đặc trưng lệch nhất.

**Phần 4 — Dịch chuyển phân phối Train so với Test**
Đo bằng cùng công cụ SMD. Trả lời câu hỏi: mô hình huấn luyện trên dữ liệu quan sát có áp được lên dữ liệu RCT không.

**Phần 5 — Cấu trúc tương quan**
`corrwith` giữa từng đặc trưng với `W` và với `Y` để nhận diện confounder. Tìm các cặp tương quan cao và các cặp tương quan tuyệt đối bằng 1 (cột thừa tuyến tính mà phép băm không bắt được). Heatmap.

**Phần 6 — Uplift theo phân khúc, không dùng mô hình**
Chia khách hàng theo phân vị của từng đặc trưng, tính chênh lệch tỉ lệ chuyển đổi kèm khoảng tin cậy, **bắt buộc trên tập RCT**. Chứng minh hiệu ứng không đồng nhất trước khi đụng tới bất kỳ mô hình nào.

**Phần 7 — Tổng hợp & bàn giao**
Bảng tổng hợp kết quả và ghi file JSON cho notebook 02.

**Nhả ra:** biểu đồ trong `reports/figures/`, file `artifacts/eda_columns_to_drop.json`

**Xong khi:** notebook chạy hết không lỗi, có kết luận rõ về mức độ bias và danh sách cột loại bỏ được tính ra từ dữ liệu chứ không gõ tay.

---

## 4. Notebook 02 — Feature Engineering & Data Split

**Nạp vào:** 2 file CSV gốc

### Các module

**1. Làm sạch**
Bỏ `f70` (hằng số), 8 cột trùng lặp hoàn toàn và 5 cột nhị phân mang tín hiệu trùng hoặc gần trùng → còn 69 đặc trưng gốc.

**2. Đặc trưng dẫn xuất**
Tạo 7 đặc trưng theo dòng: ba tỉ lệ, hai tương tác, số đặc trưng khác 0 trong nhóm top-10 và tổng ba cờ nhị phân độc lập. Ghi rõ công thức từng cái trong markdown, vì sau này Feature Contract cần mô tả lại cho DE.

**3. Chia 4 tập**
Theo thiết kế ở mục 2. In bảng kiểm tra tỉ lệ `is_treat` và `label` của cả 4 tập.

**4. Kiểm tra Positivity / Overlap**
Huấn luyện một mô hình propensity đơn giản, vẽ phân phối propensity score của hai nhóm. Nếu hai phân phối gần như không chồng lấn thì giả định positivity bị vi phạm và ước lượng CATE sẽ không đáng tin. Chốt ngưỡng cắt `[0.01, 0.99]` cho các bước sau.

**Nhả ra:** `dataset/train.parquet`, `dataset/val.parquet`, `dataset/rct_select.parquet`, `dataset/rct_holdout.parquet`

**Xong khi:** 4 file parquet tồn tại, tỉ lệ treat/label giữa Train và Val lệch nhau dưới 0,1%.

---

## 5. Notebook 03 — Hyperparameter Tuning & MLflow

**Nạp vào:** `train.parquet`, `val.parquet`

Notebook này chạy lâu (vài giờ) nên tách riêng — chạy xong một lần rồi thôi, không phải chạy lại mỗi khi chỉnh biểu đồ ở notebook 04.

### Các module

**1. Hàm chấm điểm trên tập Val**
Khoảng 15 dòng: tính DR-score bằng cross-fitting 5 fold trên tập Val (mô hình propensity + hai mô hình outcome), rồi trả về sai số bình phương giữa CATE dự đoán và DR-score. Càng thấp càng tốt.

**2. Sáu mô hình cần tinh chỉnh**

| Mô hình | Vai trò |
|---|---|
| S-Learner | Baseline, để có mốc so sánh |
| T-Learner | Baseline |
| DR-Learner (Kennedy 2020) | Nhân vật chính thứ nhất |
| LinearDML | Hệ số diễn giải được |
| NonParamDML | Bề mặt CATE phức tạp |
| CausalForestDML | Có khoảng tin cậy hợp lệ nhờ honest splitting |

Hai baseline rất rẻ mà rất đáng có: không có mốc so sánh thì không chứng minh được tính doubly robust mang lại gì.

**3. Tinh chỉnh bằng Optuna**
10–15 trial cho mỗi mô hình, không gian tìm kiếm gồm `n_estimators`, `learning_rate`, `max_depth`, `min_child_samples`. Riêng CausalForestDML chậm hơn hẳn nên giảm số trial và lấy mẫu con, ghi rõ lý do trong notebook.

**4. Ghi vết MLflow**
Mỗi trial là một run, log đầy đủ: toàn bộ siêu tham số, DR-score trên Val, thời gian chạy, số đặc trưng, seed. Tổng khoảng 70 run. Đây chính là bảng thống kê mentor muốn nhìn — để thấy có tìm kiếm thật chứ không phải gõ tay một bộ số.

**5. Chốt tham số tốt nhất**
Với mỗi mô hình lấy trial có DR-score thấp nhất, ghi ra file.

**Nhả ra:** `mlflow.db`, `artifacts/best_params.json`

**Xong khi:** MLflow UI hiện đủ các run kèm siêu tham số, file json có đủ 6 mục.

---

## 6. Notebook 04 — Benchmark & Đánh giá cuối

**Nạp vào:** `best_params.json` + cả 4 tập parquet

### Các module

**1. Bộ metric cho tập RCT**
Định nghĩa `qini_score`, `auuc`, `ate_with_ci`, `policy_simulation`.

Phân vai rõ ràng giữa bốn nhóm metric, tránh lẫn lộn:

| Nhóm | Metric | Dùng để | Tập nào |
|---|---|---|---|
| Tinh chỉnh | DR-score MSE | Chọn siêu tham số | Val (có bias) |
| **Xếp hạng — chính** | **Qini coefficient** | Chọn quán quân | RCT-select |
| Nghiệp vụ | Uplift@10/20/30%, voucher tiết kiệm | Đưa lên slide | RCT-holdout |
| Kiểm tra | ATE + khoảng tin cậy | Xác nhận không lệch hệ thống | RCT-holdout |

**Metric chính là Qini coefficient**, vì nó đo đúng thứ bài toán cần — khả năng *xếp hạng* ai đáng phát voucher — chứ không phải độ chính xác của giá trị CATE tuyệt đối, thứ mà thực tế không bao giờ kiểm chứng được.

**2. Kiểm chứng metric — làm trước khi dùng**
Ba phép thử ngược:

- Cho điểm ngẫu nhiên → Qini phải xấp xỉ 0
- `ate_with_ci` trên tập RCT phải khớp với hiệu hai tỉ lệ chuyển đổi tính tay
- Cho điểm hoàn hảo (dùng chính nhãn thật) → Qini phải cao hơn hẳn ngẫu nhiên

Nếu một trong ba phép thử sai thì công thức sai, phải sửa trước khi đi tiếp. Rất ít người tự kiểm chứng metric của mình — đây là điểm dễ ghi điểm với mentor.

**2b. Bootstrap khoảng tin cậy cho Qini — bắt buộc**

EDA cho thấy tín hiệu trong bộ dữ liệu này rất yếu: ATE trên toàn tập RCT chỉ `0,374 pp` với khoảng tin cậy `±0,169 pp`, tức khoảng 2,2 lần sai số chuẩn. Tập RCT-holdout chỉ còn một nửa nên sai số còn nở thêm khoảng 1,4 lần.

Hệ quả: **chênh lệch Qini giữa các mô hình rất dễ nằm trọn trong vùng nhiễu**. Nếu một mô hình đạt 0,041 và mô hình khác đạt 0,039 thì đó gần như chắc chắn là ngẫu nhiên.

Vì vậy mọi con số Qini báo cáo đều phải kèm khoảng tin cậy bootstrap — lấy mẫu lại có hoàn lại 1.000 lần trên tập RCT, tính Qini mỗi lần, lấy phân vị 2,5% và 97,5%. Báo cáo dạng `Qini = 0,041 [0,028 – 0,055]`.

Nếu khoảng tin cậy của các mô hình chồng lấn nhau, kết luận trung thực là **"các phương pháp không khác biệt có ý nghĩa thống kê trên bộ dữ liệu này"**. Đó là một kết luận nghiên cứu hợp lệ, và tốt hơn nhiều so với việc tuyên bố quán quân dựa trên chênh lệch ở chữ số thứ ba.

**3. Huấn luyện lại 6 mô hình**
Dùng tham số tốt nhất, huấn luyện trên Train + Val gộp lại (đã tinh chỉnh xong thì không cần giữ Val riêng nữa).

**4. Chọn quán quân trên RCT-select**
Đo AUUC và Qini của cả 6 mô hình. Chọn ra một mô hình tốt nhất.

**5. Đánh giá cuối trên RCT-holdout — chạy một lần**
Bảng benchmark đầy đủ: Qini **kèm khoảng tin cậy bootstrap**, AUUC, Uplift@20%, ATE kèm khoảng tin cậy, thời gian huấn luyện. Vẽ đường Cumulative Gain của cả 6 mô hình cùng đường baseline ngẫu nhiên.

**6. Mô phỏng chính sách**
Nếu chỉ phát voucher cho nhóm 10% / 20% / 30% có CATE cao nhất thì tỉ lệ chuyển đổi tăng bao nhiêu, và tiết kiệm được bao nhiêu voucher so với phát đại trà. Đây là con số hội đồng hiểu ngay.

**7. Feature Importance & Ablation**
Lấy độ quan trọng từ tầng cuối của DR-Learner, dùng `importance_type='gain'` chứ không dùng mặc định — mặc định đếm số lần chia nhánh nên thiên vị đặc trưng liên tục nhiều giá trị, khiến đặc trưng nhị phân bị đánh giá thấp oan. Chốt top 25, huấn luyện lại và so sánh với bản đầy đủ.

**8. Phân tích nhóm Persuadables**
Nhóm 20% có CATE cao nhất khác phần còn lại ở điểm nào. Đây là câu trả lời trực tiếp cho mục tiêu nghiệp vụ mà mentor nêu: tìm người chỉ mua khi có voucher, tránh lãng phí ngân sách cho người vốn dĩ đã mua.

**9. Xuất mô hình**
`artifacts/model.pkl` kèm `metadata.json` ghi tên mô hình, danh sách đặc trưng đúng thứ tự, các metric, ngày huấn luyện.

**Nhả ra:** bảng benchmark, biểu đồ, `artifacts/model.pkl`, `artifacts/metadata.json`

**Xong khi:** ba phép thử metric đều qua, bảng benchmark đầy đủ 6 mô hình, có kết luận rõ mô hình nào thắng và thắng nhờ đâu.

---

## 7. Tài liệu kèm theo

**`docs/causal_inference_note.md`** — bản ghi chú lý thuyết mentor dặn từ tuần 1:

- Khái niệm nền: potential outcomes, ATE, CATE, confounder
- Ba giả định cốt lõi: unconfoundedness, positivity, SUTVA — và dữ liệu này thỏa mãn tới đâu
- Hai nhánh uplift modeling, mỗi nhánh sinh ra để giải quyết vấn đề gì
- Vì sao chọn DR-Learner và DML: Neyman-orthogonality, cross-fitting, tính chất doubly robust
- Thư viện sử dụng và lý do

---

## 8. Thứ tự thực hiện & thời gian

| Bước | Nội dung | Thời gian |
|---|---|---|
| 1 | Xóa 3 notebook cũ, reset `mlflow.db` | 15 phút |
| 2 | Notebook 01 — EDA | 0,5 ngày |
| 3 | Notebook 02 — Feature Engineering & Split | 0,5 ngày |
| 4 | Notebook 04 phần metric + 3 phép kiểm chứng | 0,5 ngày |
| 5 | Notebook 03 — Tuning (phần lớn là chờ máy chạy) | 1,5 ngày |
| 6 | Notebook 04 phần còn lại — Benchmark | 1 ngày |
| 7 | `docs/causal_inference_note.md` | 0,5 ngày |
| 8 | Slide báo cáo | 0,5 ngày |

**Tổng: khoảng 5 ngày làm việc.**

Lưu ý thứ tự: bước 4 (viết và kiểm chứng metric) làm **trước** bước 5 (tuning), dù metric nằm ở notebook 04. Lý do là nếu công thức metric sai thì toàn bộ kết quả tuning phải bỏ đi chạy lại. Trong lúc chờ máy chạy tuning ở bước 5 thì viết tài liệu ở bước 7.

---

## 9. Những gì sẽ xóa

- `notebooks/01_exploratory_data_analysis.ipynb`, `02_feature_engineering_preprocessing.ipynb`, `03_cate_models_benchmark.ipynb` — làm lại từ đầu
- `notebooks/create_cate_notebook.py`, `create_modular_notebooks.py` — script sinh notebook cũ
- `mlflow.db` — các run trong đó không log siêu tham số nào, giữ lại không dùng được
- `implementation_plan.md` — thay bằng tài liệu này
- `dataset/processed_trainset.parquet`, `processed_testset.parquet` — sẽ sinh lại theo cách chia mới

**Giữ nguyên:** `dataset/full_trainset.csv`, `dataset/full_testset.csv`, `DESCN.pdf`, `mentor-webinar/`, `project-information.md`
