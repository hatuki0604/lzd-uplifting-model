"""Suy ngưỡng quyết định phát voucher từ đường cong Qini.

Script độc lập: chỉ ĐỌC artifact đã có, không huấn luyện lại, không sửa notebook.

Bài toán: mô hình cho ra `uplift_score` liên tục, nhưng chiến dịch cần quyết định
nhị phân phát / không phát. Cắt ở một con số tự đặt (0.02) là không có căn cứ, vì
điểm của DR-Learner không được hiệu chuẩn — thang tuyệt đối của nó không khớp với
thang uplift thật.

Cách làm ở đây:

1. Dựng đường cong Qini trên ``rct_select`` — đúng tập đã dùng để ra quyết định ở
   04b, nên không đụng vào ``rct_holdout``.
2. Đường cong tăng rồi giảm: thêm người đáng phát thì tăng, thêm người không phản
   ứng thì đi ngang, thêm "sleeping dog" thì giảm. Đỉnh của nó là tỉ lệ dân số nên
   nhắm tới.
3. Đỉnh thô rất nhiễu nên bootstrap lấy sai số tại đỉnh, rồi áp quy tắc 1-SE: chọn
   k NHỎ NHẤT mà đường cong còn nằm trong 1 SE của đỉnh. Ổn định hơn và tiết kiệm
   ngân sách hơn ``argmax``.
4. Đổi k sang ngưỡng điểm bằng phân vị, rồi ghép với ngưỡng hòa vốn kinh tế
   ``chi_phí_voucher / lợi_nhuận_mỗi_đơn``. Ngưỡng cuối là max của hai cái.

Chạy:

    python scripts/chon_nguong.py
    python scripts/chon_nguong.py --chi-phi-voucher 15000 --loi-nhuan-don 40000
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

try:                                   # numpy 2.x đổi tên trapz
    TRAPZ = np.trapezoid
except AttributeError:                 # pragma: no cover
    TRAPZ = np.trapz

GOC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(GOC, 'artifacts')
DATASET = os.path.join(GOC, 'dataset')
SEED = 42


# ---------------------------------------------------------------------------
# Metric — chép nguyên từ notebook 04c để script chạy độc lập, cùng seed nên
# cho ra đúng con số của notebook.
# ---------------------------------------------------------------------------
def _sap_xep(y, w, score, seed=SEED):
    n = len(y)
    rng = np.random.RandomState(seed)
    order = np.lexsort((rng.rand(n), -np.asarray(score, dtype=float)))
    y_, w_ = np.asarray(y)[order], np.asarray(w)[order]
    return (np.cumsum(w_), np.cumsum(1 - w_),
            np.cumsum(y_ * w_), np.cumsum(y_ * (1 - w_)))


def qini_curve(y, w, score, seed=SEED):
    """(x = tỉ lệ dân số được nhắm, g = số đơn tăng thêm trên mỗi người)."""
    nt, nc, yt, yc = _sap_xep(y, w, score, seed)
    n = len(y)
    ty_le = np.divide(nt, nc, out=np.zeros(n), where=nc > 0)
    return np.arange(1, n + 1) / n, (yt - yc * ty_le) / n


def gain_curve(y, w, score, seed=SEED):
    n = len(y)
    nt, nc, yt, yc = _sap_xep(y, w, score, seed)
    r_t = np.divide(yt, nt, out=np.zeros(n), where=nt > 0)
    r_c = np.divide(yc, nc, out=np.zeros(n), where=nc > 0)
    return np.arange(1, n + 1) / n, (r_t - r_c) * (nt + nc) / n


def oracle_score(y, w):
    return np.where(np.asarray(w) == 1, y, -np.asarray(y)).astype(float)


def _dien_tich_tren_duong_cheo(x, g):
    return float(TRAPZ(g, x) - 0.5 * g[-1])


def qini_score(y, w, score, seed=SEED):
    tran = _dien_tich_tren_duong_cheo(*qini_curve(y, w, oracle_score(y, w), seed))
    thuc = _dien_tich_tren_duong_cheo(*qini_curve(y, w, score, seed))
    return thuc / tran if tran > 0 else np.nan


def auuc_score(y, w, score, seed=SEED):
    x, g = gain_curve(y, w, score, seed)
    xo, go = gain_curve(y, w, oracle_score(y, w), seed)
    tran = _dien_tich_tren_duong_cheo(xo, go)
    return _dien_tich_tren_duong_cheo(x, g) / tran if tran > 0 else np.nan


def ate_with_ci(y, w, alpha=0.05):
    y = np.asarray(y, dtype=float)
    w = np.asarray(w)
    y1, y0 = y[w == 1], y[w == 0]
    if len(y1) == 0 or len(y0) == 0:
        return None
    p1, p0 = y1.mean(), y0.mean()
    se = np.sqrt(p1 * (1 - p1) / len(y1) + p0 * (1 - p0) / len(y0))
    return {'ate': float(p1 - p0), 'se': float(se),
            'ci_thap': float(p1 - p0 - 1.96 * se),
            'ci_cao': float(p1 - p0 + 1.96 * se),
            'n_treat': int(len(y1)), 'n_control': int(len(y0))}


# ---------------------------------------------------------------------------
# Chọn ngưỡng
# ---------------------------------------------------------------------------
def chon_nguong(y, w, score, seed=SEED, n_boot=300):
    """Đỉnh đường Qini + quy tắc 1-SE -> tỉ lệ nhắm và ngưỡng điểm."""
    y = np.asarray(y)
    w = np.asarray(w)
    score = np.asarray(score, dtype=float)
    n = len(y)

    x, g = qini_curve(y, w, score, seed)
    i_dinh = int(np.argmax(g))
    k_argmax, cao_dinh = float(x[i_dinh]), float(g[i_dinh])

    # Sai số của độ cao đỉnh + phân bố của chính vị trí đỉnh
    rng = np.random.RandomState(seed)
    cao_boot = np.empty(n_boot)
    k_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        xb, gb = qini_curve(y[idx], w[idx], score[idx], seed)
        j = int(np.argmax(gb))
        cao_boot[b] = gb[j]
        k_boot[b] = xb[j]

    se_dinh = float(np.std(cao_boot))
    k_ci = [float(v) for v in np.percentile(k_boot, [2.5, 97.5])]

    # Quy tắc 1-SE: k nhỏ nhất còn "ngang đỉnh"
    trong_1se = np.where(g >= cao_dinh - se_dinh)[0]
    k_chon = float(x[trong_1se[0]])

    # Đổi tỉ lệ -> ngưỡng điểm. Có thể trùng điểm nên ghi lại tỉ lệ thực tế.
    nguong = float(np.quantile(score, 1 - k_chon))
    k_thuc_te = float((score >= nguong).mean())

    return {
        'k_argmax': k_argmax,
        'k_chon': k_chon,
        'k_thuc_te_sau_lam_tron': k_thuc_te,
        'k_ci_95_bootstrap': k_ci,
        'nguong_qini': nguong,
        'cao_dinh_don_tren_nguoi': cao_dinh,
        'se_dinh_bootstrap': se_dinh,
        'dinh_tach_duoc_khoi_nhieu': bool(cao_dinh > 2 * se_dinh),
        'n_boot': n_boot,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--chi-phi-voucher', type=float, default=None,
                    help='Chi phí một voucher (VND). Bỏ trống thì bỏ qua ngưỡng kinh tế.')
    ap.add_argument('--loi-nhuan-don', type=float, default=None,
                    help='Lợi nhuận biên một đơn hàng (VND).')
    ap.add_argument('--budget', type=int, default=None,
                    help='Số voucher tối đa của chiến dịch. Cắt top-K sau ngưỡng.')
    ap.add_argument('--n-boot', type=int, default=300)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out', default=os.path.join(ARTIFACTS, 'nguong_quyet_dinh.json'))
    args = ap.parse_args()

    khoa = json.load(open(os.path.join(ARTIFACTS, 'decisions_locked.json')))
    quan_quan = khoa['quan_quan']

    rct = pd.read_parquet(os.path.join(DATASET, 'rct_select.parquet'))
    cate = np.load(os.path.join(ARTIFACTS, '04a_cate_select.npz'), allow_pickle=True)
    score = np.asarray(cate[quan_quan], dtype=float)
    y = rct['label'].to_numpy()
    w = rct['is_treat'].to_numpy()

    if len(score) != len(y):
        raise SystemExit(f'Lệch cỡ: cate={len(score)} vs rct_select={len(y)}')

    print(f'Mô hình quán quân : {quan_quan}')
    print(f'Tập ra quyết định : rct_select ({len(y):,} người) — holdout KHÔNG mở')
    print(f'Mã băm quyết định : {khoa["ma_bam"]}')

    ate = ate_with_ci(y, w)
    print(f'\nATE trên rct_select : {ate["ate"]*100:+.4f} pp '
          f'[{ate["ci_thap"]*100:+.4f} – {ate["ci_cao"]*100:+.4f}]')
    print(f'Hệ số Qini          : {qini_score(y, w, score, args.seed):.5f}')
    print(f'Hệ số AUUC          : {auuc_score(y, w, score, args.seed):.5f}')

    kq = chon_nguong(y, w, score, seed=args.seed, n_boot=args.n_boot)

    print('\n--- Đường cong Qini ---')
    print(f'Đỉnh thô (argmax)        : k = {kq["k_argmax"]*100:.1f}%')
    print(f'KTC 95% của vị trí đỉnh  : [{kq["k_ci_95_bootstrap"][0]*100:.1f}% – '
          f'{kq["k_ci_95_bootstrap"][1]*100:.1f}%]')
    print(f'Độ cao đỉnh              : {kq["cao_dinh_don_tren_nguoi"]*1000:.4f} đơn/1000 người')
    print(f'Sai số bootstrap của đỉnh: {kq["se_dinh_bootstrap"]*1000:.4f} đơn/1000 người')
    print(f'Đỉnh tách khỏi nhiễu     : {"CÓ" if kq["dinh_tach_duoc_khoi_nhieu"] else "KHÔNG"}')
    print(f'\nk chọn (quy tắc 1-SE)    : {kq["k_chon"]*100:.1f}%')
    print(f'Ngưỡng điểm từ Qini      : {kq["nguong_qini"]:.6f}')

    # Ngưỡng hòa vốn kinh tế
    nguong_kinh_te = None
    if args.chi_phi_voucher is not None and args.loi_nhuan_don:
        nguong_kinh_te = args.chi_phi_voucher / args.loi_nhuan_don
        print(f'Ngưỡng hòa vốn kinh tế   : {nguong_kinh_te:.6f} '
              f'({args.chi_phi_voucher:,.0f} / {args.loi_nhuan_don:,.0f})')

    nguong_cuoi = kq['nguong_qini']
    nguon = 'qini_1se'
    if nguong_kinh_te is not None and nguong_kinh_te > nguong_cuoi:
        nguong_cuoi, nguon = nguong_kinh_te, 'hoa_von_kinh_te'

    chon = score >= nguong_cuoi
    print(f'\n=== NGƯỠNG CUỐI: {nguong_cuoi:.6f} (nguồn: {nguon}) ===')
    print(f'Số người được phát : {int(chon.sum()):,} / {len(score):,} '
          f'({chon.mean()*100:.1f}%)')

    # Kiểm chứng trên chính rct_select: nhóm được chọn có uplift thật cao hơn không
    kd = {}
    for ten, mask in [('Được phát', chon), ('Không phát', ~chon)]:
        r = ate_with_ci(y[mask], w[mask])
        if r is None:
            continue
        kd[ten] = r
        print(f'  {ten:<11}: n={int(mask.sum()):>7,}  '
              f'uplift thực đo {r["ate"]*100:+.4f} pp '
              f'[{r["ci_thap"]*100:+.4f} – {r["ci_cao"]*100:+.4f}]')

    if len(kd) == 2:
        a, b = kd['Được phát'], kd['Không phát']
        d = a['ate'] - b['ate']
        se_d = np.sqrt(a['se']**2 + b['se']**2)
        print(f'  Chênh lệch : {d*100:+.4f} pp ± {1.96*se_d*100:.4f} pp  '
              f'-> khác 0 có ý nghĩa: {"CÓ" if abs(d) > 1.96*se_d else "KHÔNG"}')

    if args.budget:
        print(f'\nSau ràng buộc ngân sách {args.budget:,} voucher: '
              f'lấy top-{min(args.budget, int(chon.sum())):,} trong nhóm đã qua ngưỡng')

    ket_qua = {
        'quan_quan': quan_quan,
        'tap_ra_quyet_dinh': 'rct_select',
        'n': int(len(y)),
        'seed': args.seed,
        'thoi_diem': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ma_bam_quyet_dinh_04b': khoa['ma_bam'],
        'qini': float(qini_score(y, w, score, args.seed)),
        'auuc': float(auuc_score(y, w, score, args.seed)),
        'ate_rct_select': ate,
        **kq,
        'nguong_kinh_te': nguong_kinh_te,
        'chi_phi_voucher': args.chi_phi_voucher,
        'loi_nhuan_don': args.loi_nhuan_don,
        'nguong_cuoi': float(nguong_cuoi),
        'nguon_nguong_cuoi': nguon,
        'ti_le_duoc_phat': float(chon.mean()),
        'budget': args.budget,
        'kiem_chung_rct_select': {k: v for k, v in kd.items()},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(ket_qua, f, ensure_ascii=False, indent=2)
    print(f'\nĐã ghi: {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
