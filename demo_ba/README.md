# Dashboard demo quyết định voucher cho BA

Dashboard trả lời bốn câu hỏi theo đúng thứ tự nghiệp vụ:

1. Model có tạo ra nhóm khách phản ứng với voucher tốt hơn không?
2. Uplift đo được của từng nhóm Top% chắc chắn đến đâu?
3. Với chi phí, biên lợi nhuận và ngân sách hiện tại, nên chọn Top% nào?
4. Nhóm được chọn khác phần khách còn lại ở những feature nào, và API sẽ xử lý
   từng khách ra sao?

Dashboard nằm trong repo DS. Nó chỉ gọi FastAPI đã có của repo DE; không copy
serving code, không sửa Redis và không tạo tín hiệu realtime giả trong giao diện.

## Những phần đã triển khai

### 1. Bằng chứng hiệu quả campaign trên RCT holdout

`offline_policy.json` được sinh lại từ ba artifact thật:

- `dataset/rct_holdout.parquet`: 90.835 khách, có treatment/control ngẫu nhiên;
- `artifacts/model_booster.txt`: model DRLearner đã bàn giao;
- `artifacts/feature_contract.json`: đúng thứ tự 76 feature model nhận vào.

Model chấm uplift dự đoán cho từng khách, xếp hạng giảm dần với seed 42 rồi lấy
đúng `round(Top% × 90.835)` khách. Uplift của mỗi nhóm không phải trung bình
prediction; nó được đo lại bằng:

```text
uplift nhóm = conversion rate treatment − conversion rate control
đơn tăng thêm = số khách trong nhóm × uplift nhóm
```

Dashboard so sánh Top 10%, 20%, 30%, 50% và phát đại trà. Các point estimate
được tái tạo khớp Bước 37 của `notebooks/04_benchmark_evaluation.ipynb`.

### 2. Khoảng tin cậy 95%

Mỗi Top% hiển thị uplift point estimate, cỡ mẫu treatment/control và KTC 95%
theo Wald interval cho chênh lệch hai tỷ lệ. KTC thể hiện độ bất định do chỉ quan
sát một mẫu RCT:

- khoảng hoàn toàn lớn hơn 0: uplift của nhóm có bằng chứng dương ở mức 5%;
- khoảng chứa 0: chưa đủ bằng chứng kết luận uplift của riêng nhóm khác 0;
- khoảng càng rộng: ước lượng càng kém chính xác.

KTC của lợi nhuận được suy ra bằng cách thay uplift thấp/cao vào cùng công thức
kinh tế. Đây không phải cam kết doanh thu.

### 3. Bộ tính hiệu quả kinh doanh

BA có thể nhập trực tiếp năm biến:

- chi phí mỗi voucher thực sự được sử dụng;
- tỷ lệ khách sử dụng voucher;
- biên lợi nhuận trên mỗi đơn tăng thêm;
- ngân sách campaign tối đa;
- chi phí campaign cố định.

Dashboard tính từng policy theo công thức:

```text
số voucher dùng kỳ vọng = số khách mục tiêu × tỷ lệ sử dụng
chi phí biến đổi = số voucher dùng kỳ vọng × chi phí mỗi voucher
tổng chi phí = chi phí biến đổi + chi phí cố định
biên lợi nhuận tăng thêm = đơn tăng thêm × biên lợi nhuận mỗi đơn
lãi ròng = biên lợi nhuận tăng thêm − tổng chi phí
ROI = lãi ròng / tổng chi phí
```

Policy được đề xuất là policy có lãi ròng kỳ vọng cao nhất trong ngân sách. Hệ
thống luôn so với phương án **không chạy campaign**, có lãi bằng 0; do đó nếu
mọi policy đều lỗ hoặc đều vượt ngân sách, dashboard sẽ đề xuất không chạy.

Giá trị mặc định chỉ là giả định demo: voucher 20.000 đồng, redemption 10%, biên
lợi nhuận 200.000 đồng/đơn, ngân sách 100 triệu và chi phí cố định bằng 0. Với
bộ giả định này, Top 20% có point estimate lợi nhuận cao nhất. Điều đó **không
có nghĩa Top 20% là policy cố định hoặc tối ưu cho doanh nghiệp**.

### 4. “Nhóm khách này là ai?”

Với mỗi Top 10/20/30/50%, dashboard hiển thị 10 feature có độ chênh chuẩn hóa
(SMD) lớn nhất giữa nhóm mục tiêu và phần khách còn lại:

```text
SMD = (trung bình nhóm mục tiêu − trung bình phần còn lại) / độ lệch chuẩn gộp
```

SMD dương nghĩa là feature trung bình cao hơn trong nhóm mục tiêu; SMD âm nghĩa
là thấp hơn. Dữ liệu nguồn chỉ có tên ẩn danh như `f28`, `f26` hoặc feature dẫn
xuất như `fe_inter_f9_f26`, nên dashboard không tự gán ý nghĩa kinh doanh chưa
được cung cấp.

SMD chỉ mô tả nhóm model đã chọn. Nó không phải feature importance và không
chứng minh feature đó gây ra uplift.

### 5. API live cho 10 khách synthetic

Phần dưới dashboard gọi `/ready`, `/store/info` và `/campaign/decide` từ FastAPI
DE để minh họa xếp hạng và sáu action ở cấp từng khách:

| API action | Nhãn BA |
|---|---|
| `SEND_VOUCHER` | Gửi voucher |
| `WAIT_FOR_INTENT` | Chờ thêm ý định mua |
| `SUPPRESS_ALREADY_PURCHASED` | Không gửi vì khách đã mua |
| `NOT_SELECTED_BUDGET` | Ngoài nhóm được xét ưu tiên |
| `SKIP_NON_POSITIVE_UPLIFT` | Voucher không được dự đoán mang lại hiệu quả |
| `NO_MODEL_SCORE` | Không chấm được điểm |

Cohort live cố định `U0000000`–`U0000009` để demo lặp lại ổn định. Top-K tại
đây là số slot API xét trong 10 khách, không phải Top% policy trên 90.835 khách
RCT và cũng không bảo đảm từng slot sẽ được gửi voucher. Policy không backfill
slot bị `WAIT` hoặc `SUPPRESS`.

## Cách chạy

### Terminal 1 — chuẩn bị stack DE

```bash
cd "/Users/hatrungkien/MacDrive/GreenSM/nhung-lala/LZD"
make track-b-online-demo
```

Lệnh này chuẩn bị dữ liệu synthetic, Kafka/Redis và FastAPI tại
`http://localhost:18000`. Nên chạy lại ngay trước buổi demo vì tín hiệu realtime
có cửa sổ một giờ.

### Terminal 2 — chạy dashboard DS

```bash
cd "/Users/hatrungkien/MacDrive/GreenSM/final-lazada"
.venv/bin/uv sync --frozen --extra demo
./demo_ba/run_dashboard.sh
```

Mở `http://127.0.0.1:8501` và giữ Terminal đang chạy. Luôn chạy từ repo root
hoặc dùng script trên để tránh lỗi import package `demo_ba`.

## Sinh lại bằng chứng offline

Khi model, feature contract hoặc RCT holdout thay đổi, chạy:

```bash
cd "/Users/hatrungkien/MacDrive/GreenSM/final-lazada"
.venv/bin/python -m demo_ba.generate_policy_evidence
```

Script ghi lại `demo_ba/offline_policy.json`, kèm SHA-256 của ba artifact nguồn,
KTC 95%, cỡ mẫu và feature profile. Không sửa JSON bằng tay.

## Kiểm thử

```bash
cd "/Users/hatrungkien/MacDrive/GreenSM/final-lazada"
.venv/bin/python -m unittest discover -s tests -v
```

Test hiện bao phủ tính nhất quán RCT evidence, KTC/cỡ mẫu/profile, công thức chi
phí và ngân sách, rule chọn Top 20% dưới giả định mặc định, rule “không chạy”,
đủ sáu API action, payload/lỗi 422, độ mới realtime và artifact handoff DS → DE.

## Kịch bản trình bày BA khoảng 3 phút

1. Ở bảng RCT, nói: model xếp hạng từng người nhưng hiệu quả được đo trên cả
   nhóm bằng treatment/control; Top 10/20/30/50 chỉ là các kịch bản so sánh.
2. Chỉ vào uplift và KTC 95% để phân biệt point estimate với độ chắc chắn.
3. Nhập chi phí, redemption, margin và ngân sách thật hoặc dùng giả định demo;
   giải thích policy xanh là policy có lãi kỳ vọng cao nhất trong ngân sách.
4. Mở hồ sơ nhóm để trả lời “nhóm này khác ở feature nào”, kèm cảnh báo SMD chỉ
   mang tính mô tả và feature đang bị ẩn danh.
5. Xuống API live để minh họa cách danh sách mục tiêu được xếp hạng và cách tín
   hiệu realtime gate hành động gửi/chờ/chặn cho từng khách.

## Giới hạn cần nói rõ

- Kết quả Top% là mô phỏng policy và uplift đo trên RCT holdout, không phải kết
  quả realtime của 10 khách demo.
- KTC hiện phản ánh sai số lấy mẫu của chênh lệch hai tỷ lệ; chưa cộng thêm độ
  bất định của model và chưa điều chỉnh việc so sánh đồng thời nhiều Top%.
- Qini holdout của model có KTC 95% chứa 0. Không được tuyên bố model đã chứng
  minh khả năng xếp hạng tốt ở mức ý nghĩa 95%; nên xác nhận policy cuối bằng
  validation ngoài mẫu và A/B test mới.
- Input tài chính mặc định không phải số liệu doanh nghiệp. Policy đề xuất chỉ
  đúng dưới các giả định đang nhập.
- Feature `f*` chưa có data dictionary nghiệp vụ nên dashboard chỉ mô tả tên và
  phân phối, không tự diễn giải chúng thành hành vi khách hàng.
- Tín hiệu `rt_*` không nằm trong vector model hiện tại; nó chỉ gate thời điểm
  action của policy demo và không làm đổi uplift score.
