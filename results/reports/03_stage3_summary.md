# Stage 3 — Data Preparation Summary

**CRISP-DM Phase:** Data Preparation  
**Script:** `src/stage3_data_preparation.py`  
**Dijalankan via:** `run_pipeline.py --stage stage3`

---

## Tujuan Stage Ini

Mengubah raw dataset ILPD menjadi dataset siap-latih yang memenuhi tiga syarat:

1. **Reprodusibel** — random state terkunci, scaler params tersimpan sebagai JSON
2. **Anti-leakage** — scaler di-fit hanya pada training set
3. **Fairness-aware** — stratifikasi 4-class melindungi subgrup minoritas

---

## Langkah Preprocessing (Berurutan)

### 1. Target Encoding
| Nilai Original | Nilai Encoded |
|---|---|
| `Liver Disease` | `1` |
| `No Liver Disease` | `0` |

Kolom baru: `target` (integer binary)

---

### 2. Gender Encoding
| Nilai Original | Nilai Encoded |
|---|---|
| `Male` | `1` |
| `Female` | `0` |

Kolom baru: `Gender_encoded` (integer binary)  
**Kolom `Gender` asli tetap disimpan** di dataset untuk keperluan subgroup tracking di Stage 4.

---

### 3. Feature Engineering — `fractional_bilirubin`

```
fractional_bilirubin = DB / (TB + 1e-6)
```

**Rasional klinis:** Fractional bilirubin (rasio DB/TB) adalah indikator fungsi hati yang mapan secara medis. Berdasarkan analisis Stage 2, nilai DB median pasien perempuan sakit = median pasien sehat (0.20), sehingga rasio relatif ini diharapkan lebih diskriminatif daripada nilai absolut.

---

### 4. Log Transformation (`log1p`)

Diterapkan pada 5 fitur berikut sebelum scaling:

| Fitur | Alasan |
|---|---|
| `TB` | Right-skewed, max 75.0 vs median 1.2 |
| `DB` | Right-skewed, max 19.7 vs median 0.4 |
| `Alkphos` | Right-skewed, max 2110 vs median 215 |
| `Sgpt` | Sangat ekstrem, max 2000 vs median 38 |
| `Sgot` | Paling ekstrem, max 4929 vs median 46 |

Transformasi `log1p` = $\ln(x + 1)$ — aman untuk nilai 0, kompres ekor kanan distribusi.

---

### 5. Train–Test Split — Stratifikasi 4-Class

**Strategi:** Membuat key stratifikasi gabungan `Gender_Target` = 4 kelas:
- Male — Liver Disease
- Male — No Liver Disease
- Female — Liver Disease
- Female — No Liver Disease ← **subgrup terkritis (hanya 50 pasien)**

**Parameter:**
```
test_size    = 0.20
random_state = 42
```

**Hasil split:**

| Split | Gender | Target | Count | % dari Split |
|---|---|---|---|---|
| Train | Female | No Liver Disease | 40 | 8.58% |
| Train | Female | Liver Disease | 74 | 15.88% |
| Train | Male | No Liver Disease | 93 | 19.96% |
| Train | Male | Liver Disease | 259 | 55.58% |
| **Test** | **Female** | **No Liver Disease** | **10** | **8.55%** |
| Test | Female | Liver Disease | 18 | 15.38% |
| Test | Male | No Liver Disease | 24 | 20.51% |
| Test | Male | Liver Disease | 65 | 55.56% |

Proporsi `Female — No Liver Disease` identik di kedua split (8.58% vs 8.55%) — stratifikasi 4-class berhasil menjaga representasi subgrup minoritas.

---

### 6. Feature Scaling — RobustScaler

**Scaler:** `sklearn.preprocessing.RobustScaler`  
**Formula:** $x' = \frac{x - \text{median}}{\text{IQR}}$

**Mengapa bukan StandardScaler:**
StandardScaler menggunakan mean dan standar deviasi, yang sangat sensitif terhadap outlier ekstrem (Sgot max = 4929). RobustScaler berbasis median dan IQR sehingga tidak terdistorsi oleh nilai-nilai ekstrem ini.

**Scaler parameters (fit dari training set):**

| Fitur | Median (center) | IQR (scale) |
|---|---|---|
| Age | 45.0000 | 25.0000 |
| TB *(after log1p)* | 0.6931 | 0.7205 |
| DB *(after log1p)* | 0.2624 | 0.6825 |
| Alkphos *(after log1p)* | 5.3519 | 0.5285 |
| Sgpt *(after log1p)* | 3.6109 | 0.9769 |
| Sgot *(after log1p)* | 3.7842 | 1.2150 |
| TP | 6.5000 | 1.5000 |
| ALB | 3.1000 | 1.1000 |
| A/G Ratio | 0.9000 | 0.4000 |
| fractional_bilirubin | 0.3333 | 0.2119 |

Parameters tersimpan lengkap di `data/processed/scaler_meta.json`.

---

## Fitur Final (11 fitur)

| # | Fitur | Tipe | Keterangan |
|---|---|---|---|
| 1 | Age | numeric | Usia pasien |
| 2 | TB | numeric | Total Bilirubin (log1p) |
| 3 | DB | numeric | Direct Bilirubin (log1p) |
| 4 | Alkphos | numeric | Alkaline Phosphotase (log1p) |
| 5 | Sgpt | numeric | Alamine Aminotransferase (log1p) |
| 6 | Sgot | numeric | Aspartate Aminotransferase (log1p) |
| 7 | TP | numeric | Total Protein |
| 8 | ALB | numeric | Albumin |
| 9 | A/G Ratio | numeric | Albumin & Globulin Ratio |
| 10 | fractional_bilirubin | engineered | DB / (TB + ε) — rasio klinis |
| 11 | Gender_encoded | categorical_binary | Male=1, Female=0 |

---

## Artifact yang Dihasilkan

### Processed Data
| File | Deskripsi |
|---|---|
| `data/processed/train.csv` | 466 baris — scaled, siap training |
| `data/processed/test.csv` | 117 baris — scaled, siap evaluasi |
| `data/processed/train_raw.csv` | 466 baris — unscaled, untuk SHAP & interpretasi |
| `data/processed/test_raw.csv` | 117 baris — unscaled |
| `data/processed/scaler_meta.json` | Median & IQR per fitur untuk reproduksi |

### Tables
| File | Deskripsi |
|---|---|
| `results/tables/03_selected_features.csv` | Daftar 11 fitur final + tipe |
| `results/tables/03_split_balance.csv` | Distribusi 4-class di train & test |
| `results/tables/03_subgroup_descriptive_stats.csv` | Statistik deskriptif per Gender×Target (siap masuk paper) |

### Reports
| File | Deskripsi |
|---|---|
| `results/reports/03_data_preparation.txt` | Log langkah preprocessing |
| `results/reports/03_ai_ready_preprocessing_summary.txt` | Summary ringkas untuk Stage 4 |

---

## Catatan Penting untuk Stage 4

1. **Kolom `Gender`** (string asli) tersedia di `train.csv` dan `test.csv` — selalu gunakan kolom ini untuk semua subgroup split, bukan `Gender_encoded`.
2. **Kolom `Selector`** (target string asli) juga masih ada di dataset sebagai referensi — gunakan kolom `target` (int) untuk training.
3. **Untuk SHAP dan feature importance interpretation** — gunakan `train_raw.csv` / `test_raw.csv` agar nilai dapat diinterpretasikan dalam skala original (setelah log transform, sebelum RobustScaling).
4. **Bootstrap CI** pada subgroup metrics di Stage 4 tetap direkomendasikan mengingat `Female — No Liver Disease` test set hanya 10 sampel.
