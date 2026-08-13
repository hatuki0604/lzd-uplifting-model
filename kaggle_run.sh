#!/usr/bin/env bash
# Đẩy notebook lên Kaggle chạy, không cần mở trình duyệt.
#
#   ./kaggle_run.sh push 03      # đẩy notebook 03 lên và chạy
#   ./kaggle_run.sh status 03    # xem chạy tới đâu
#   ./kaggle_run.sh pull 03      # tải kết quả về thư mục kaggle_output/03/
#   ./kaggle_run.sh data         # cập nhật lại dataset trên Kaggle sau khi đổi file trong dataset/
set -euo pipefail

KAGGLE_USER="hatrungkienhatuki"
DATASET="${KAGGLE_USER}/lazada-uplift-data"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${2:-}" in
  03) NB="03_model_tuning.ipynb";        SLUG="lazada-03-tuning";    DEPS='' ;;
  04) NB="04_benchmark_evaluation.ipynb"; SLUG="lazada-04-benchmark"; DEPS="\"${KAGGLE_USER}/lazada-03-tuning\"" ;;
  *)  NB=""; SLUG=""; DEPS='' ;;
esac

stage() {
  local dir="$ROOT/.kaggle_stage/$SLUG"
  rm -rf "$dir"; mkdir -p "$dir"
  cp "$ROOT/notebooks/$NB" "$dir/"
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
    [ -n "$NB" ] || { echo "Dùng: $0 push 03|04"; exit 1; }
    dir="$(stage)"
    kaggle kernels push -p "$dir"
    echo
    echo "Đã đẩy lên. Theo dõi bằng:  $0 status ${2}"
    echo "Xem trực tiếp: https://www.kaggle.com/code/${KAGGLE_USER}/${SLUG}"
    ;;
  status)
    [ -n "$SLUG" ] || { echo "Dùng: $0 status 03|04"; exit 1; }
    kaggle kernels status "${KAGGLE_USER}/${SLUG}"
    ;;
  pull)
    [ -n "$SLUG" ] || { echo "Dùng: $0 pull 03|04"; exit 1; }
    out="$ROOT/kaggle_output/${2}"
    mkdir -p "$out"
    kaggle kernels output "${KAGGLE_USER}/${SLUG}" -p "$out"
    echo "Kết quả đã tải về: $out"
    ;;
  data)
    # Chỉ đẩy đúng thứ notebook 03/04 cần. KHÔNG đẩy full_trainset.csv (476 MB) và
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
    sed -n '2,8p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
