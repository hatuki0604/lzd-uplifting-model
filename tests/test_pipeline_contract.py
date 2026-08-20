"""Chốt cổng cho pipeline 4 notebook.

Bộ test này không chạy lại mô hình. Nó kiểm những bất biến mà mọi lần chạy đều
phải giữ, và là những chỗ đã từng lệch âm thầm trong các bản trước:

1. Đúng **69 đặc trưng** ở mọi artifact, và đúng danh sách 14 cột bị loại bỏ.
2. **Không còn cột `fe_*`** nào — pipeline này không tạo đặc trưng dẫn xuất.
3. **Phép chia dữ liệu không đổi**: chia lại từ CSV bằng seed 42 phải ra đúng
   `split_assignment.parquet`.
4. **Thứ tự đặc trưng** khớp nhau giữa parquet, `feature_info.json`,
   `best_params.json`, `feature_contract.json` và `metadata.json`.
5. **Prediction parity**: `model.pkl` nạp lại cho đúng các giá trị trong
   `golden_predictions.csv`, và bản `model_booster.txt` cũng vậy; số cột booster
   khớp số cột contract.

Chạy:
    uv run --frozen python -m unittest tests.test_pipeline_contract -v

Test nào thiếu artifact đầu vào thì tự bỏ qua kèm lời nhắc chạy notebook nào.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
DATA = ROOT / 'dataset'

SO_DAC_TRUNG = 69
COT_HANG_SO = ['f70']
COT_TRUNG_KHIT = ['f15', 'f32', 'f33', 'f39', 'f71', 'f72', 'f77', 'f78']
COT_NHI_PHAN_DU_THUA = ['f67', 'f69', 'f73', 'f74', 'f76']
COT_LOAI_BO = sorted(COT_HANG_SO + COT_TRUNG_KHIT + COT_NHI_PHAN_DU_THUA,
                     key=lambda c: int(c[1:]))
TAP_CON = ['train', 'val', 'rct_select', 'rct_holdout']
SEED = 42


def _doc_json(path: Path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _can_file(path: Path, notebook: str):
    if not path.exists():
        raise unittest.SkipTest(f'Thiếu {path.relative_to(ROOT)} — chạy {notebook} trước.')
    return path


def _bam_data_id(ids) -> str:
    """Đúng cách notebook 01 băm: sort theo chuỗi, nối bằng '|', sha256, lấy 16 ký tự."""
    arr = np.sort(np.asarray(ids).astype(str))
    return hashlib.sha256('|'.join(arr).encode()).hexdigest()[:16]


class TestBanGiaoEDA(unittest.TestCase):
    """artifacts/eda_columns_to_drop.json — notebook 01."""

    @classmethod
    def setUpClass(cls):
        cls.eda = _doc_json(_can_file(ART / 'eda_columns_to_drop.json', 'notebook 01'))

    def test_dung_69_dac_trung(self):
        self.assertEqual(self.eda['so_dac_trung_sach'], SO_DAC_TRUNG)
        self.assertEqual(len(self.eda['dac_trung_sach']), SO_DAC_TRUNG)

    def test_danh_sach_cot_loai_bo_dung_nhu_da_chot(self):
        self.assertEqual(self.eda['cot_hang_so'], COT_HANG_SO)
        self.assertEqual(sorted(self.eda['cot_trung_lap'], key=lambda c: int(c[1:])),
                         COT_TRUNG_KHIT)
        self.assertEqual(self.eda['cot_nhi_phan_du_thua'], COT_NHI_PHAN_DU_THUA)
        self.assertEqual(self.eda['cot_can_loai_bo'], COT_LOAI_BO)

    def test_83_tru_14_bang_69(self):
        tat_ca = [f'f{i}' for i in range(83)]
        self.assertEqual(self.eda['dac_trung_sach'],
                         [c for c in tat_ca if c not in set(COT_LOAI_BO)])

    def test_khong_co_dac_trung_dan_xuat(self):
        self.assertFalse([c for c in self.eda['dac_trung_sach'] if c.startswith('fe_')])


class TestPhepChiaKhongDoi(unittest.TestCase):
    """Phép chia phải tái lập được từ hai file CSV gốc bằng seed 42."""

    @classmethod
    def setUpClass(cls):
        cls.eda = _doc_json(_can_file(ART / 'eda_columns_to_drop.json', 'notebook 01'))
        cls.assignment = pd.read_parquet(
            _can_file(ART / 'split_assignment.parquet', 'notebook 01'))

    def test_so_dong_va_ma_bam_khop_artifact(self):
        so_dong = {k: int(v) for k, v in self.assignment['split'].value_counts().items()}
        self.assertEqual(so_dong, self.eda['phan_chia']['so_dong'])

        for ten, ma in self.eda['phan_chia']['ma_bam_data_id'].items():
            ids = self.assignment.loc[self.assignment['split'] == ten, 'data_id']
            self.assertEqual(_bam_data_id(ids), ma, f'mã băm tập {ten} lệch')

    def test_khong_co_data_id_nam_o_hai_tap(self):
        self.assertEqual(self.assignment['data_id'].duplicated().sum(), 0)

    def test_chia_lai_tu_csv_ra_dung_ket_qua_cu(self):
        from sklearn.model_selection import train_test_split

        tr_csv = _can_file(DATA / 'full_trainset.csv', 'không có dữ liệu gốc')
        te_csv = _can_file(DATA / 'full_testset.csv', 'không có dữ liệu gốc')

        cols = ['data_id', 'is_treat', 'label']
        tr = pd.read_csv(tr_csv, usecols=cols)
        te = pd.read_csv(te_csv, usecols=cols)

        def strata(df):
            return df['is_treat'].astype(str) + '_' + df['label'].astype(str)

        tr_idx, val_idx = train_test_split(tr.index, test_size=0.20,
                                           random_state=SEED, stratify=strata(tr))
        sel_idx, hold_idx = train_test_split(te.index, test_size=0.50,
                                             random_state=SEED, stratify=strata(te))

        chia_lai = {
            'train': tr.loc[tr_idx, 'data_id'],
            'val': tr.loc[val_idx, 'data_id'],
            'rct_select': te.loc[sel_idx, 'data_id'],
            'rct_holdout': te.loc[hold_idx, 'data_id'],
        }
        for ten, ids in chia_lai.items():
            self.assertEqual(_bam_data_id(ids),
                             self.eda['phan_chia']['ma_bam_data_id'][ten],
                             f'Chia lại tập {ten} ra kết quả khác — phép chia đã đổi')


class TestBanGiaoFeatureInfo(unittest.TestCase):
    """artifacts/feature_info.json + 4 file parquet — notebook 02."""

    @classmethod
    def setUpClass(cls):
        cls.eda = _doc_json(_can_file(ART / 'eda_columns_to_drop.json', 'notebook 01'))
        cls.finfo = _doc_json(_can_file(ART / 'feature_info.json', 'notebook 02'))

    def test_69_dac_trung_va_dung_thu_tu_cua_notebook_01(self):
        self.assertEqual(self.finfo['so_dac_trung'], SO_DAC_TRUNG)
        self.assertEqual(self.finfo['tat_ca_dac_trung'], self.eda['dac_trung_sach'])

    def test_khong_tao_dac_trung_dan_xuat(self):
        self.assertEqual(self.finfo['dac_trung_dan_xuat'], [])
        self.assertFalse([c for c in self.finfo['tat_ca_dac_trung'] if c.startswith('fe_')])

    def test_khong_ghi_thong_ke_nhan_cua_holdout(self):
        """rct_holdout chỉ được ghi phần thuộc thiết kế thí nghiệm.

        Tỉ lệ mua theo từng nhánh của holdout chính là ATE mà notebook 04 báo cáo;
        ghi nó ra từ notebook 02 là mở tập holdout sớm.
        """
        chia = self.finfo['chia_du_lieu']
        self.assertIsNone(chia['rct_holdout']['ti_le_mua'])
        for ten in ['train', 'val', 'rct_select']:
            self.assertIsNotNone(chia[ten]['ti_le_mua'], f'{ten}: thiếu thống kê hợp lệ')

    def test_bon_file_parquet_dung_cot_dung_thu_tu_dung_so_dong(self):
        mong_doi = ['data_id', 'is_treat', 'label'] + self.finfo['tat_ca_dac_trung']
        for ten in TAP_CON:
            path = _can_file(DATA / f'{ten}.parquet', 'notebook 02')
            df = pd.read_parquet(path)
            self.assertEqual(list(df.columns), mong_doi, f'{ten}: sai cột hoặc sai thứ tự')
            self.assertEqual(len(df), self.eda['phan_chia']['so_dong'][ten])
            self.assertEqual(int(df.isna().sum().sum()), 0, f'{ten}: có giá trị khuyết')


class TestBanGiaoTuning(unittest.TestCase):
    """artifacts/best_params.json — notebook 03."""

    @classmethod
    def setUpClass(cls):
        cls.finfo = _doc_json(_can_file(ART / 'feature_info.json', 'notebook 02'))
        cls.tuning = _doc_json(_can_file(ART / 'best_params.json', 'notebook 03'))

    def test_tinh_chinh_tren_dung_69_dac_trung(self):
        cau_hinh = self.tuning['cau_hinh']
        self.assertEqual(cau_hinh['n_features'], SO_DAC_TRUNG)
        self.assertEqual(cau_hinh['feature_names'], self.finfo['tat_ca_dac_trung'])

    def test_du_sau_mo_hinh(self):
        self.assertEqual(
            sorted(self.tuning['best_params']),
            sorted(['SLearner', 'TLearner', 'DRLearner',
                    'LinearDML', 'NonParamDML', 'CausalForestDML']))

    def test_che_do_full_phai_co_66_trial(self):
        if self.tuning['run_mode'] != 'full':
            self.skipTest('best_params.json đang là kết quả smoke test')
        self.assertEqual(self.tuning['cau_hinh']['tong_trial'], 66)
        self.assertEqual(
            sum(v['n_trials'] for v in self.tuning['best_params'].values()), 66)


class TestBanGiaoModel(unittest.TestCase):
    """model.pkl / feature_contract.json / metadata.json / golden — notebook 04."""

    @classmethod
    def setUpClass(cls):
        cls.finfo = _doc_json(_can_file(ART / 'feature_info.json', 'notebook 02'))
        cls.contract = _doc_json(_can_file(ART / 'feature_contract.json', 'notebook 04'))
        cls.meta = _doc_json(_can_file(ART / 'metadata.json', 'notebook 04'))
        cls.ten_cot = [d['ten'] for d in sorted(
            cls.contract['dac_trung'], key=lambda d: d['thu_tu_dua_vao_mo_hinh'])]

    def test_contract_khong_chua_cot_dan_xuat(self):
        self.assertFalse([t for t in self.ten_cot if t.startswith('fe_')],
                         'feature_contract.json còn cột fe_*')

    def test_contract_chi_gom_cot_goc_da_lam_sach(self):
        hop_le = set(self.finfo['tat_ca_dac_trung'])
        self.assertTrue(set(self.ten_cot).issubset(hop_le))
        self.assertFalse(set(self.ten_cot) & set(COT_LOAI_BO))

    def test_thu_tu_dua_vao_mo_hinh_lien_tuc_tu_0(self):
        thu_tu = [d['thu_tu_dua_vao_mo_hinh'] for d in self.contract['dac_trung']]
        self.assertEqual(sorted(thu_tu), list(range(len(thu_tu))))
        self.assertEqual(self.contract['so_cot_mo_hinh_nhan_vao'], len(self.ten_cot))

    def test_contract_khop_metadata(self):
        self.assertEqual(self.ten_cot, self.meta['dac_trung'])
        self.assertEqual(self.meta['so_dac_trung'], len(self.ten_cot))
        self.assertEqual(self.meta['so_dac_trung_day_du'], SO_DAC_TRUNG)
        self.assertEqual(self.meta['dac_trung_day_du'], self.finfo['tat_ca_dac_trung'])
        # thứ tự trong contract phải là thứ tự cột gốc, không phải thứ tự theo độ quan trọng
        self.assertEqual(self.ten_cot,
                         [c for c in self.finfo['tat_ca_dac_trung'] if c in set(self.ten_cot)])

    def test_moi_muc_co_du_truong_bat_buoc(self):
        for d in self.contract['dac_trung']:
            for truong in ('ten', 'thu_tu_dua_vao_mo_hinh', 'gia_tri_mac_dinh', 'khoang'):
                self.assertIn(truong, d)
            self.assertTrue(np.isfinite(d['gia_tri_mac_dinh']))

    def test_so_cot_booster_khop_contract(self):
        info = self.meta.get('booster')
        if not info:
            self.skipTest(f'{self.meta["ten_mo_hinh"]} không có bản booster dạng text')
        import lightgbm as lgb

        bu = len(info.get('cot_bo_sung', []))
        for t in info['files']:
            path = _can_file(ART / t['file'], 'notebook 04')
            booster = lgb.Booster(model_file=str(path))
            self.assertEqual(booster.num_feature(), t['so_cot_dau_vao'])
            self.assertEqual(booster.num_feature(), len(self.ten_cot) + bu,
                             f"{t['file']}: số cột booster không khớp contract")

    def test_prediction_parity_voi_golden_predictions(self):
        golden_path = _can_file(ART / 'golden_predictions.csv', 'notebook 04')
        model_path = _can_file(ART / 'model.pkl', 'notebook 04')
        rct_path = _can_file(DATA / 'rct_select.parquet', 'notebook 02')

        golden = pd.read_csv(golden_path, float_precision='round_trip')
        rct = pd.read_parquet(rct_path)

        vi_tri = pd.Series(np.arange(len(rct)), index=rct['data_id'].astype(str).values)
        dong = vi_tri.reindex(golden['data_id'].astype(str).values).values
        self.assertFalse(np.isnan(dong).any(), 'golden có data_id không nằm trong rct_select')

        X = rct[self.ten_cot].values.astype(np.float32)[dong.astype(int)]

        # cloudpickle đóng gói class kèm code object, mà cấu trúc code object không tương
        # thích chéo bản Python. Artifact đóng gói trên Kaggle (3.12) không nạp được bằng
        # môi trường khoá của repo (3.10). Đó không phải lỗi của bản bàn giao — model_booster.txt
        # sinh ra chính vì lý do này, và test kế bên kiểm parity bằng nó.
        py_goi = self.meta.get('serialization', {}).get('python_dong_goi')
        py_dang_chay = platform.python_version()
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except (TypeError, ValueError) as exc:
            if py_goi and py_goi != py_dang_chay:
                self.skipTest(
                    f'model.pkl đóng gói bằng Python {py_goi}, đang chạy {py_dang_chay} '
                    f'-> không nạp được ({type(exc).__name__}). Parity được kiểm bằng '
                    'model_booster.txt ở test kế bên.')
            raise

        cate = np.asarray(model.predict_cate(X), dtype=np.float64)

        # Dung sai do notebook 04 chốt: 0 tuyệt đối với mô hình LightGBM (tất định tới
        # từng bit), 1e-12 với estimator EconML (BLAS có thể đổi thứ tự phép cộng khi
        # mảng đầu vào nằm khác chỗ trong bộ nhớ).
        tol = float(self.meta.get('golden_predictions', {}).get('dung_sai', 0.0))
        lech = float(np.abs(cate - golden['cate'].values).max())
        self.assertLessEqual(
            lech, tol,
            f'model.pkl không tái lập golden_predictions.csv (lệch {lech:.2e} > {tol:g})')

    def test_booster_cho_dung_ket_qua_nhu_model_pkl(self):
        info = self.meta.get('booster')
        if not info:
            self.skipTest(f'{self.meta["ten_mo_hinh"]} không có bản booster dạng text')
        import lightgbm as lgb

        golden = pd.read_csv(_can_file(ART / 'golden_predictions.csv', 'notebook 04'), float_precision='round_trip')
        rct = pd.read_parquet(_can_file(DATA / 'rct_select.parquet', 'notebook 02'))
        vi_tri = pd.Series(np.arange(len(rct)), index=rct['data_id'].astype(str).values)
        dong = vi_tri.reindex(golden['data_id'].astype(str).values).values.astype(int)
        X = rct[self.ten_cot].values.astype(np.float32)[dong]

        bs = [lgb.Booster(model_file=str(ART / t['file'])) for t in info['files']]
        ten_mo_hinh = self.meta['ten_mo_hinh']
        if ten_mo_hinh == 'DRLearner':
            cate = bs[0].predict(X)
        elif ten_mo_hinh == 'SLearner':
            n = len(X)
            cate = (bs[0].predict(np.column_stack([X, np.ones(n)]))
                    - bs[0].predict(np.column_stack([X, np.zeros(n)])))
        elif ten_mo_hinh == 'TLearner':
            cate = bs[0].predict(X) - bs[1].predict(X)
        else:
            self.skipTest(f'{ten_mo_hinh}: chưa định nghĩa cách ghép CATE từ booster')

        np.testing.assert_array_equal(
            np.asarray(cate, dtype=np.float64), golden['cate'].values,
            'model_booster.txt không cho cùng kết quả với model.pkl')

    def test_chuoi_04a_04b_04c_khop_nhau(self):
        """Ba mảnh của notebook 04 phải cùng nói về một lần chạy.

        Ghép nhầm output của lần chạy khác là lỗi im lặng: mọi con số vẫn in ra
        bình thường, chỉ có điều chúng thuộc về hai mô hình khác nhau.
        """
        chuoi = self.meta.get('chuoi_notebook')
        if not chuoi:
            self.skipTest('metadata.json đến từ bản notebook 04 gộp, không có chuỗi 04a/04b/04c')
        if '04a_run_id' not in chuoi:
            self.skipTest('bản bàn giao đến từ chuỗi 07->08->09 — xem '
                          'test_chuoi_07_08_09_khop_nhau')

        tm_path, sm_path = ART / '04a_train_meta.json', ART / '04b_select_meta.json'
        if not (tm_path.exists() and sm_path.exists()):
            self.skipTest('không có 04a_train_meta.json / 04b_select_meta.json để đối chiếu')

        train_meta, select_meta = _doc_json(tm_path), _doc_json(sm_path)

        self.assertEqual(select_meta['run_id_04a'], train_meta['run_id'])
        self.assertEqual(chuoi['04a_run_id'], train_meta['run_id'])
        self.assertEqual(chuoi['04b_run_id'], select_meta['run_id_04b'])
        self.assertEqual(train_meta['ma_bam_dac_trung'], select_meta['ma_bam_dac_trung'])
        self.assertEqual(select_meta['ma_bam_quyet_dinh'],
                         self.meta['quyet_dinh_da_khoa']['ma_bam'])
        self.assertEqual(select_meta['quan_quan'], self.meta['ten_mo_hinh'])
        self.assertEqual(select_meta['dac_trung_ban_giao'], self.ten_cot)

    def test_so_dong_huan_luyen_duoc_ghi_lai_day_du(self):
        """Số dòng mỗi mô hình học phải đi kèm bảng benchmark.

        Mô hình học trên ít dữ liệu hơn thì không so ngang hàng được; giấu con số đó
        đi là biến một bảng so sánh phương pháp thành một bảng so sánh ngân sách tính toán.
        """
        dl = self.meta.get('du_lieu_huan_luyen', {})
        so_dong = dl.get('so_dong_moi_mo_hinh')
        if not so_dong:
            self.skipTest('metadata.json đến từ bản notebook 04 gộp')

        self.assertEqual(set(so_dong), set(self.meta['metric_rct_holdout']))
        self.assertIn('cung_co_mau', dl)
        self.assertEqual(dl['cung_co_mau'], len(set(so_dong.values())) == 1)

    def test_file_khoa_04b_tu_nhat_quan(self):
        """decisions_locked.json phải luôn tự nhất quán, dù bản đang giao là bản nào.

        Đây là hồ sơ của lần mở holdout duy nhất. Nó phải đọc được và băm đúng kể cả
        khi bản bàn giao đã đổi sang chuỗi khác.
        """
        khoa = _doc_json(_can_file(ART / 'decisions_locked.json', 'notebook 04'))
        goi = {k: v for k, v in khoa.items() if k not in ('thoi_diem_khoa', 'ma_bam')}
        tinh_lai = hashlib.sha256(
            json.dumps(goi, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
        self.assertEqual(tinh_lai, khoa['ma_bam'])

    def test_quyet_dinh_da_duoc_khoa_truoc_khi_mo_holdout(self):
        """Bản bàn giao từ chuỗi 04 phải đúng là bản đã khóa trước khi mở holdout."""
        if 'cong_quyet_dinh' in self.meta:
            self.skipTest('bản bàn giao đến từ chuỗi 07->08->09, không đi qua 04b — '
                          'xem test_chuoi_07_08_09_khop_nhau')

        khoa = _doc_json(_can_file(ART / 'decisions_locked.json', 'notebook 04'))
        self.assertEqual(khoa['quan_quan'], self.meta['ten_mo_hinh'])
        self.assertEqual(khoa['dac_trung_ban_giao'], self.ten_cot)
        self.assertEqual(khoa['k_chot'], len(self.ten_cot))
        self.assertEqual(self.meta['quyet_dinh_da_khoa']['ma_bam'], khoa['ma_bam'])

    def test_nguong_quyet_dinh_khong_bi_cu(self):
        """`nguong_quyet_dinh.json` phải nói về CHÍNH mô hình đang bàn giao.

        Ngưỡng được suy từ phân vị điểm CATE, mà thang điểm là của riêng từng mô hình.
        Đổi mô hình rồi giữ lại ngưỡng cũ là lỗi IM LẶNG: file vẫn đọc được, con số vẫn
        hợp lệ, chỉ có điều nó là phân vị trên một phân bố điểm không còn tồn tại.

        Notebook 10 phải chạy SAU notebook 09 — đánh số chỉ là quy ước, test này mới là
        thứ chặn được.
        """
        ng_path = ART / 'nguong_quyet_dinh.json'
        if not ng_path.exists():
            self.skipTest('chưa chạy notebook 10 — không có nguong_quyet_dinh.json')
        ng = _doc_json(ng_path)

        self.assertEqual(ng['quan_quan'], self.meta['ten_mo_hinh'],
                         'ngưỡng tính cho mô hình khác với mô hình đang bàn giao')

        so_cot = ng.get('so_cot_mo_hinh')
        if so_cot is None:
            self.fail('nguong_quyet_dinh.json thiếu so_cot_mo_hinh — bản cũ, chạy lại notebook 10')
        self.assertEqual(so_cot, len(self.ten_cot),
                         f'ngưỡng tính trên {so_cot} cột nhưng contract đang giao '
                         f'{len(self.ten_cot)} cột — chạy lại notebook 10 sau notebook 09')

        # Bản bàn giao đi qua cổng 08 thì ngưỡng phải mang đúng mã băm cổng đó.
        cong_meta = self.meta.get('cong_quyet_dinh')
        if cong_meta:
            self.assertEqual(ng.get('ma_bam_cong_08'), cong_meta['ma_bam'],
                             'ngưỡng thuộc về một lần chạy cổng khác')

        # Ngưỡng phải nằm trong khoảng điểm mà mô hình thật sự sinh ra.
        self.assertGreater(ng['ti_le_duoc_phat'], 0.0)
        self.assertLessEqual(ng['ti_le_duoc_phat'], 1.0)

    def test_chuoi_07_08_09_khop_nhau(self):
        """Bản bàn giao từ chuỗi 07->08->09 phải khớp cổng quyết định đã chốt.

        Chuỗi này KHÔNG đi qua rct_holdout, nên chỗ dựa của nó là cổng quyết định mà
        notebook 08 khóa bằng mã băm TRƯỚC khi có bất kỳ con số nào. Test này thay thế
        vai trò mà `decisions_locked.json` đảm nhiệm ở chuỗi 04 — không phải nới lỏng,
        mà là kiểm đúng thứ chuỗi này dựa vào.
        """
        cong_meta = self.meta.get('cong_quyet_dinh')
        if not cong_meta:
            self.skipTest('bản bàn giao đến từ chuỗi 04, không có cổng quyết định')

        so_sanh = _doc_json(_can_file(ART / 'so_sanh_k30.json', 'notebook 08'))
        xh = _doc_json(_can_file(ART / 'xep_hang_val.json', 'notebook 07'))
        bp = _doc_json(_can_file(ART / 'best_params_k30.json', 'notebook 07'))
        cong = so_sanh['cong_quyet_dinh']

        # 1. Bằng chứng phải là bản full và cổng phải ĐẠT
        for ten, doc in (('so_sanh_k30', so_sanh), ('xep_hang_val', xh),
                         ('best_params_k30', bp)):
            self.assertEqual(doc['run_mode'], 'full', f'{ten}.json không phải bản full')
        self.assertTrue(cong['ket_qua']['dat'], 'cổng quyết định không đạt')

        # 2. Mã băm cổng phải tính đúng từ chính định nghĩa cổng, và khớp metadata
        tinh_lai = hashlib.sha256(
            json.dumps(cong['dinh_nghia'], sort_keys=True,
                       ensure_ascii=False).encode('utf-8')).hexdigest()[:16]
        self.assertEqual(tinh_lai, cong['ma_bam'],
                         'định nghĩa cổng đã bị sửa sau khi chốt')
        self.assertEqual(cong_meta['ma_bam'], cong['ma_bam'])

        # 3. Tập cột đang giao phải đúng là tập của nhánh ứng viên đã được chấm
        ma_uv = cong['dinh_nghia']['nhanh_ung_vien']
        nhanh = next(n for n in so_sanh['nhanh'] if n['ma'] == ma_uv)
        self.assertEqual(nhanh['dac_trung'], self.ten_cot,
                         f'contract khác tập cột của nhánh {ma_uv} mà notebook 08 đã chấm')
        self.assertEqual(xh['dac_trung_o_muc_chot'], self.ten_cot)
        self.assertEqual(bp['cau_hinh']['feature_names'], self.ten_cot)

        # 4. Siêu tham số đang giao phải đúng bộ đã tinh chỉnh cho tập cột đó
        self.assertEqual(self.meta['sieu_tham_so'],
                         bp['best_params']['DRLearner']['params'])

        # 5. Giới hạn phải được ghi rõ, không để trống
        gh = self.meta.get('gioi_han', {})
        self.assertTrue(gh.get('chua_do_tren_rct_holdout'),
                        'metadata phải ghi rõ bản này chưa đo trên rct_holdout')
        self.assertNotIn('metric_rct_holdout', self.meta,
                         'bản chuỗi 09 không được mang số holdout — nó chưa từng được đo ở đó')


if __name__ == '__main__':
    unittest.main(verbosity=2)
