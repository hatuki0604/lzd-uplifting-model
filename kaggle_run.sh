#!/usr/bin/env bash
# Đẩy notebook lên Kaggle chạy, không cần mở trình duyệt.
#
#   ./kaggle_run.sh push 03      # tinh chỉnh siêu tham số
#   ./kaggle_run.sh push 04a     # huấn luyện 6 mô hình      (cần output của 03)
#   ./kaggle_run.sh push 04b     # chọn quán quân + đặc trưng (cần 03 và 04a)
#   ./kaggle_run.sh push 04c     # holdout + bàn giao         (cần 04a và 04b)
#   ./kaggle_run.sh push 05      # quét số đặc trưng nhiều seed (cần 03 và 04b)
#   ./kaggle_run.sh push 07      # xếp hạng đặc trưng trên Val + tinh chỉnh 30 cột (cần 03, 04b, 05)
#   ./kaggle_run.sh push 08      # so 30 cột với 69 cột trên rct_select        (cần 03, 05, 07)
#   ./kaggle_run.sh push 09      # đóng gói bản 30 cột                          (cần 03, 04b, 07, 08)
#
#   ./kaggle_run.sh status 04a   # xem chạy tới đâu
#   ./kaggle_run.sh pull 04a     # tải kết quả về kaggle_output/04a/
#   ./kaggle_run.sh status 05    # xem vòng quét đặc trưng
#   ./kaggle_run.sh pull 05      # tải JSON + biểu đồ của notebook 05
#   ./kaggle_run.sh data         # cập nhật dataset trên Kaggle sau khi đổi file trong dataset/
#
# Notebook 04 được tách làm ba mảnh vì mỗi kernel Kaggle chỉ có 12 giờ, mà riêng
# CausalForestDML trên toàn bộ dữ liệu đã ăn gần hết ngần ấy. Tách ra thì mỗi mảnh có
# quota riêng, hỏng chỗ nào chạy lại chỗ đó, và rct_holdout chỉ tồn tại trong 04c.
#
# Chạy THỨ TỰ: 03 -> 04a -> 04b -> 04c -> 05 -> 07 -> 08 -> 09. Mỗi bước phải xong (status = complete) rồi mới đẩy bước sau,
# vì Kaggle gắn output của kernel trước qua kernel_sources.
#
# Mặc định đẩy lên là chạy chế độ ĐẦY ĐỦ. Muốn thử nhanh trên Kaggle:
#   KAGGLE_SMOKE=1 ./kaggle_run.sh push 05
set -euo pipefail

KAGGLE_USER="hatrungkienhatuki"
DATASET="${KAGGLE_USER}/lazada-uplift-data"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

K03="\"${KAGGLE_USER}/lazada-03-tuning\""
K04A="\"${KAGGLE_USER}/lazada-04a-train\""
K04B="\"${KAGGLE_USER}/lazada-04b-select\""
K05="\"${KAGGLE_USER}/lazada-05-features\""
K07="\"${KAGGLE_USER}/lazada-07-val-rank\""
K08="\"${KAGGLE_USER}/lazada-08-so-sanh\""

case "${2:-}" in
  03)  NB="03_model_tuning.ipynb";      SLUG="lazada-03-tuning";   DEPS='' ;;
  04a) NB="04a_train_models.ipynb";     SLUG="lazada-04a-train";   DEPS="${K03}" ;;
  04b) NB="04b_select_model.ipynb";     SLUG="lazada-04b-select";  DEPS="${K03}, ${K04A}" ;;
  04c) NB="04c_holdout_report.ipynb";   SLUG="lazada-04c-holdout"; DEPS="${K04A}, ${K04B}" ;;
  05)  NB="05_quet_so_dac_trung.ipynb"; SLUG="lazada-05-features"; DEPS="${K03}, ${K04B}" ;;
  07)  NB="07_xep_hang_dac_trung_val.ipynb"; SLUG="lazada-07-val-rank";  DEPS="${K03}, ${K04B}, ${K05}" ;;
  08)  NB="08_so_sanh_k30.ipynb";            SLUG="lazada-08-so-sanh";   DEPS="${K03}, ${K05}, ${K07}" ;;
  09)  NB="09_ban_giao_k30.ipynb";           SLUG="lazada-09-ban-giao";  DEPS="${K03}, ${K04B}, ${K07}, ${K08}" ;;
  *)   NB=""; SLUG=""; DEPS='' ;;
esac

SMOKE="${KAGGLE_SMOKE:-0}"

stage() {
  local dir="$ROOT/.kaggle_stage/$SLUG"
  rm -rf "$dir"; mkdir -p "$dir"
  cp "$ROOT/notebooks/$NB" "$dir/"

  # Notebook mặc định chạy chế độ nhanh để lỡ quên thì không đốt hàng giờ CPU. Trên Kaggle
  # thì ngược lại: đã đẩy lên là để chạy thật. Chèn một ô lệnh đặt biến môi trường ngay đầu
  # notebook thay vì bắt người dùng sửa tay — sửa tay là chỗ dễ quên nhất.
  SMOKE="$SMOKE" python3 - "$dir/$NB" <<'PYEOF'
import json, os, sys
p = sys.argv[1]
smoke = os.environ['SMOKE']
nb = json.load(open(p, encoding='utf-8'))
cell = {
    'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
    'source': [
        "# Ô lệnh do kaggle_run.sh chèn vào — đặt chế độ chạy trước khi notebook đọc biến này.\n",
        "import os\n",
        f"os.environ['SMOKE_TEST'] = '{smoke}'\n",
        f"print('SMOKE_TEST =', os.environ['SMOKE_TEST'], "
        "'->', 'thử nhanh' if os.environ['SMOKE_TEST'] == '1' else 'chạy đầy đủ')\n",
    ],
}
nb['cells'].insert(0, cell)
json.dump(nb, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
PYEOF
  cat > "$dir/kernel-metadata.json" <<EOF
{
  "id": "${KAGGLE_USER}/${SLUG}",
  "title": "${SLUG}",
  "code_file": "${NB}",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": ["${DATASET}"],
  "competition_sources": [],
  "kernel_sources": [${DEPS}]
}
EOF
  echo "$dir"
}

case "${1:-}" in
  push)
    [ -n "$NB" ] || { echo "Dùng: $0 push 03|04a|04b|04c|05|07|08|09"; exit 1; }
    dir="$(stage)"
    echo "Notebook : $NB"
    echo "Kernel   : ${KAGGLE_USER}/${SLUG}"
    echo "Phụ thuộc: ${DEPS:-không}"
    echo "Chế độ   : $([ "$SMOKE" = "1" ] && echo 'THỬ NHANH (KAGGLE_SMOKE=1)' || echo 'ĐẦY ĐỦ')"
    echo
    kaggle kernels push -p "$dir"
    echo
    echo "Đã đẩy lên. Theo dõi bằng:  $0 status ${2}"
    echo "Xem trực tiếp: https://www.kaggle.com/code/${KAGGLE_USER}/${SLUG}"
    ;;
  status)
    [ -n "$SLUG" ] || { echo "Dùng: $0 status 03|04a|04b|04c|05"; exit 1; }
    kaggle kernels status "${KAGGLE_USER}/${SLUG}"
    ;;
  pull)
    [ -n "$SLUG" ] || { echo "Dùng: $0 pull 03|04a|04b|04c|05"; exit 1; }
    out="$ROOT/kaggle_output/${2}"
    mkdir -p "$out"
    kaggle kernels output "${KAGGLE_USER}/${SLUG}" -p "$out"
    echo "Kết quả đã tải về: $out"
    ;;
  data)
    # Chỉ đẩy đúng thứ notebook 03/04/05 cần. KHÔNG đẩy full_trainset.csv (476 MB) và
    # full_testset.csv (98 MB) — hai file đó chỉ notebook 01 dùng, đẩy lên vừa lâu vừa vô ích.
    dir="$ROOT/.kaggle_stage/dataset"
    rm -rf "$dir"; mkdir -p "$dir"
    cp "$ROOT/dataset/train.parquet"       "$dir/"
    cp "$ROOT/dataset/val.parquet"         "$dir/"
    cp "$ROOT/dataset/rct_select.parquet"  "$dir/"
    cp "$ROOT/dataset/rct_holdout.parquet" "$dir/"
    cp "$ROOT/artifacts/feature_info.json" "$dir/"
    # KHÔNG đẩy best_params.json vào đây. Nó là đầu ra của notebook 03, không phải dữ liệu.
    # Đẩy kèm thì notebook 04 trên Kaggle sẽ vớ phải bản cũ trong dataset thay vì đọc
    # kết quả tươi từ kernel notebook 03 — sai mà không báo lỗi.
    cat > "$dir/dataset-metadata.json" <<EOF
{
  "title": "lazada-uplift-data",
  "id": "${DATASET}",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
    echo "Sắp đẩy lên $(du -sh "$dir" | cut -f1):"
    ls -la "$dir"
    echo
    read -rp "Xác nhận đẩy lên Kaggle? [y/N] " ok
    [ "$ok" = "y" ] || { echo "Đã huỷ."; exit 0; }
    kaggle datasets version -p "$dir" -m "cap nhat $(date +%F)" --dir-mode zip
    ;;
  *)
    sed -n '2,26p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
