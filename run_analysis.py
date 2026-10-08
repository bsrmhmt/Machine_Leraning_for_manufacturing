"""
Otomatik Makine Öğrenmesi & Çok Amaçlı Optimizasyon Sistemi
===========================================================
Kullanım:
  1. Veri dosyanızı (CSV veya Excel) bu klasöre koyun.
  2. Aşağıdaki AYARLAR bölümünü düzenleyin.
  3. Terminalde çalıştırın:
       python run_analysis.py
  4. Tüm grafikler + PDF otomatik olarak oluşturulur.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI gerektirmeyen backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import sys
import datetime
import warnings
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# =====================================================================
# AYARLAR - SADECE BURAYI DEĞİŞTİRİN
# =====================================================================
DATA_FILE = "veri.xlsx"          # Veri dosyanızın adı (CSV veya Excel)
INPUT_COLUMNS = []               # Boş bırakırsanız sizden sorar
TARGET_COLUMNS = []              # Boş bırakırsanız sizden sorar
OUTPUT_PDF = "ML_Analiz_Raporu.pdf"
NSGA_TRIALS = 1000               # Optimizasyon deneme sayısı
TEST_SIZE = 0.2
RANDOM_STATE = 42
# =====================================================================


def load_data(filepath):
    """Veriyi oku ve temizle."""
    print(f"\n📂 Veri okunuyor: {filepath}")
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    
    # Sayısal dönüşüm
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    
    print(f"   ✅ {df.shape[0]} satır, {df.shape[1]} sütun yüklendi.")
    
    # Eksik veri kontrolü
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"   ⚠️  {missing} eksik değer bulundu, temizleniyor...")
        df = df.dropna()
    
    return df


def ask_columns(df, input_cols, target_cols):
    """Eğer sütunlar belirtilmemişse kullanıcıdan sor."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not input_cols or not target_cols:
        print("\n📋 Sayısal sütunlar:")
        for i, col in enumerate(numeric_cols):
            print(f"   [{i}] {col}")
        
        if not input_cols:
            inp = input("   Girdi sütun numaralarını girin (virgülle): ")
            input_cols = [numeric_cols[int(x.strip())] for x in inp.split(",")]
        
        if not target_cols:
            tgt = input("   Hedef sütun numaralarını girin (virgülle): ")
            target_cols = [numeric_cols[int(x.strip())] for x in tgt.split(",")]
    
    print(f"\n   🟢 Girdi: {input_cols}")
    print(f"   🔴 Hedef: {target_cols}")
    return input_cols, target_cols


def get_models():
    """5 regresyon modeli."""
    return {
        "MLR": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "SVR": SVR(kernel='rbf', C=100, epsilon=0.1),
        "GPR": GaussianProcessRegressor(
            kernel=Matern(nu=2.5) + WhiteKernel(), random_state=42, n_restarts_optimizer=3
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=42, objective='reg:squarederror', verbosity=0
        )
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, scaler):
    """Tüm modelleri eğit ve değerlendir."""
    models = get_models()
    results = {}
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred_train = model.predict(X_train_sc)
        y_pred_test = model.predict(X_test_sc)
        
        n_cv = min(3, len(X_train_sc))
        if n_cv >= 2:
            cv_scores = cross_val_score(model, X_train_sc, y_train, cv=n_cv, scoring='neg_root_mean_squared_error')
            cv_mean = -cv_scores.mean()
            cv_std = cv_scores.std()
        else:
            cv_mean = 0
            cv_std = 0

        results[name] = {
            "model": model,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "cv_rmse_mean": cv_mean,
            "cv_rmse_std": cv_std,
        }
    return results


def calculate_critic_weights(df_targets):
    """CRITIC yöntemiyle hedef ağırlıkları."""
    if len(df_targets) < 3:
        n = len(df_targets.columns)
        return pd.Series([1.0/n]*n, index=df_targets.columns)
    
    norm_df = pd.DataFrame()
    for col in df_targets.columns:
        c_max = df_targets[col].max()
        c_min = df_targets[col].min()
        if c_max == c_min:
            norm_df[col] = 1.0
        else:
            norm_df[col] = (c_max - df_targets[col]) / (c_max - c_min)
    
    std = norm_df.std()
    corr = norm_df.corr()
    conflict = (1 - corr).sum(axis=1)
    C = std * conflict
    
    total = C.sum()
    if total == 0:
        n = len(df_targets.columns)
        return pd.Series([1.0/n]*n, index=df_targets.columns)
    return C / total


def apply_codas(df_targets, weights, tau=0.02):
    """CODAS sıralaması."""
    norm_df = pd.DataFrame()
    for col in df_targets.columns:
        min_val = df_targets[col].min()
        norm_df[col] = min_val / (df_targets[col] + 1e-9)
    
    w_norm = norm_df * weights
    nis = w_norm.min()
    
    E = np.sqrt(((w_norm - nis)**2).sum(axis=1))
    T = np.abs(w_norm - nis).sum(axis=1)
    
    n = len(df_targets)
    H = np.zeros((n, n))
    E_vals = E.values
    T_vals = T.values
    for i in range(n):
        for j in range(n):
            if i != j:
                tau_func = 1 if abs(E_vals[i] - E_vals[j]) > tau else 0
                H[i,j] = (E_vals[i] - E_vals[j]) + (tau_func * (T_vals[i] - T_vals[j]))
    return np.sum(H, axis=1)


def fig_to_buf(fig):
    """Figure'ü PNG buffer'a çevir."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


# =====================================================================
# PDF OLUŞTURUCU
# =====================================================================

class PDFReport:
    def __init__(self, filepath):
        self.filepath = filepath
        self.c = canvas.Canvas(filepath, pagesize=A4)
        self.w, self.h = A4
        self.y = self.h - 50
        self.page_num = 1
    
    def new_page(self):
        self.c.showPage()
        self.y = self.h - 50
        self.page_num += 1
    
    def title(self, text, size=18):
        if self.y < 100:
            self.new_page()
        self.c.setFont("Helvetica-Bold", size)
        self.c.drawString(40, self.y, text)
        self.y -= size + 10
    
    def subtitle(self, text, size=13):
        if self.y < 80:
            self.new_page()
        self.c.setFont("Helvetica-Bold", size)
        self.c.drawString(40, self.y, text)
        self.y -= size + 8
    
    def text(self, text, size=10):
        if self.y < 40:
            self.new_page()
        self.c.setFont("Helvetica", size)
        self.c.drawString(50, self.y, text)
        self.y -= size + 5
    
    def add_image(self, img_buf, width=500, height=300):
        if self.y - height < 30:
            self.new_page()
        img_buf.seek(0)
        img = ImageReader(img_buf)
        self.c.drawImage(img, 45, self.y - height, width=width, height=height, preserveAspectRatio=True)
        self.y -= height + 15
    
    def add_table(self, headers, rows):
        """Basit tablo çizici."""
        if self.y < 100:
            self.new_page()
        
        n_cols = len(headers)
        col_width = min(75, (self.w - 80) / n_cols)
        
        self.c.setFont("Helvetica-Bold", 8)
        for i, h in enumerate(headers):
            self.c.drawString(40 + i * col_width, self.y, str(h)[:12])
        self.y -= 15
        
        self.c.setFont("Helvetica", 7)
        for row in rows:
            if self.y < 40:
                self.new_page()
                self.c.setFont("Helvetica", 7)
            for i, val in enumerate(row):
                self.c.drawString(40 + i * col_width, self.y, str(val)[:12])
            self.y -= 12
    
    def save(self):
        self.c.save()
        print(f"\n📄 PDF kaydedildi: {os.path.abspath(self.filepath)}")


# =====================================================================
# ANA FONKSİYON
# =====================================================================

def run():
    print("=" * 60)
    print("  OTOMATIK ML & COK AMACLI OPTIMIZASYON SISTEMI")
    print("=" * 60)

    # 1. Veri yükle
    if not os.path.exists(DATA_FILE):
        print(f"\n❌ HATA: '{DATA_FILE}' bulunamadı!")
        print(f"   Lütfen veri dosyanızı şu klasöre koyun:")
        print(f"   {os.getcwd()}")
        sys.exit(1)
    
    df = load_data(DATA_FILE)
    print(f"\n   Sütunlar: {df.columns.tolist()}")
    
    input_cols, target_cols = ask_columns(df, INPUT_COLUMNS, TARGET_COLUMNS)
    
    # PDF başlat
    pdf = PDFReport(OUTPUT_PDF)
    pdf.title("Otomatik Makine Ogrenmesi Raporu")
    pdf.text(f"Tarih: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    pdf.text(f"Veri: {DATA_FILE} ({df.shape[0]} satir, {df.shape[1]} sutun)")
    pdf.text(f"Girdi: {', '.join(input_cols)}")
    pdf.text(f"Hedef: {', '.join(target_cols)}")
    pdf.y -= 10

    # ============ BOLUM 1: VERI KESFI ============
    print("\n" + "="*60)
    print("  BOLUM 1: VERI KESFI")
    print("="*60)
    
    # Korelasyon
    fig_corr, ax = plt.subplots(figsize=(8, 6))
    corr = df[input_cols + target_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f', ax=ax, linewidths=0.5)
    ax.set_title("Pearson Korelasyon Matrisi")
    buf_corr = fig_to_buf(fig_corr)
    print("   ✅ Korelasyon haritası oluşturuldu.")
    
    pdf.new_page()
    pdf.title("1. Veri Kesfi")
    pdf.subtitle("Korelasyon Isi Haritasi")
    pdf.add_image(buf_corr, width=480, height=350)
    
    # Dağılım
    all_cols = input_cols + target_cols
    n = len(all_cols)
    fig_dist, axes = plt.subplots(1, n, figsize=(4*n, 3.5))
    if n == 1:
        axes = [axes]
    for i, col in enumerate(all_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='steelblue')
        axes[i].set_title(col, fontsize=9)
    fig_dist.tight_layout()
    buf_dist = fig_to_buf(fig_dist)
    print("   ✅ Dağılım grafikleri oluşturuldu.")
    
    pdf.new_page()
    pdf.subtitle("Dagilim Grafikleri")
    pdf.add_image(buf_dist, width=520, height=250)

    # Scatter
    for t in target_cols:
        ni = len(input_cols)
        fig_sc, axes_sc = plt.subplots(1, ni, figsize=(4*ni, 3.5))
        if ni == 1:
            axes_sc = [axes_sc]
        for j, inp in enumerate(input_cols):
            axes_sc[j].scatter(df[inp], df[t], alpha=0.7, edgecolors='k', s=50)
            axes_sc[j].set_xlabel(inp)
            axes_sc[j].set_ylabel(t)
            axes_sc[j].set_title(f"{inp} -> {t}", fontsize=9)
        fig_sc.tight_layout()
        buf_sc = fig_to_buf(fig_sc)
        pdf.new_page()
        pdf.subtitle(f"Scatter: {t}")
        pdf.add_image(buf_sc, width=520, height=260)
    print("   ✅ Scatter grafikleri oluşturuldu.")

    # ============ BOLUM 2: MODEL EGITIMI ============
    print("\n" + "="*60)
    print("  BOLUM 2: MODEL EGITIMI")
    print("="*60)
    
    all_results = {}
    best_models = {}
    master_scaler = StandardScaler()
    master_scaler.fit(df[input_cols].values)
    
    for t_idx, target_col in enumerate(target_cols):
        print(f"\n   [{t_idx+1}/{len(target_cols)}] Hedef: {target_col}")
        
        X = df[input_cols].values
        y = df[target_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        scaler = StandardScaler()
        results = train_and_evaluate(X_train, X_test, y_train, y_test, scaler)
        all_results[target_col] = results
        
        # En iyi model
        best_name = max(results.items(), key=lambda x: x[1]["test_r2"])[0]
        best_models[target_col] = results[best_name]["model"]
        
        # Konsola yazdır
        print(f"   {'Model':<18} {'Train R²':>10} {'Test R²':>10} {'Test RMSE':>10} {'Test MAE':>10}")
        print(f"   {'-'*58}")
        for mname, mdata in results.items():
            marker = " ★" if mname == best_name else ""
            print(f"   {mname:<18} {mdata['train_r2']:>10.4f} {mdata['test_r2']:>10.4f} {mdata['test_rmse']:>10.4f} {mdata['test_mae']:>10.4f}{marker}")
        
        # PDF: Metrik tablosu
        pdf.new_page()
        pdf.title(f"2. Model Egitimi - {target_col}")
        pdf.subtitle(f"En iyi: {best_name} (Test R2={results[best_name]['test_r2']:.4f})")
        
        headers = ["Model", "Train R2", "Test R2", "Train RMSE", "Test RMSE", "Train MAE", "Test MAE", "CV RMSE"]
        rows = []
        for mname, mdata in results.items():
            rows.append([
                mname[:15],
                f"{mdata['train_r2']:.4f}",
                f"{mdata['test_r2']:.4f}",
                f"{mdata['train_rmse']:.4f}",
                f"{mdata['test_rmse']:.4f}",
                f"{mdata['train_mae']:.4f}",
                f"{mdata['test_mae']:.4f}",
                f"{mdata['cv_rmse_mean']:.4f}",
            ])
        pdf.add_table(headers, rows)
        pdf.y -= 10
        
        # Gerçek vs Tahmin grafiği
        n_models = len(results)
        fig_pred, axes_pred = plt.subplots(1, n_models, figsize=(4*n_models, 3.5))
        if n_models == 1:
            axes_pred = [axes_pred]
        for k, (mname, mdata) in enumerate(results.items()):
            ax = axes_pred[k]
            ax.scatter(y_test, mdata["y_pred_test"], alpha=0.7, edgecolors='k', s=40)
            lims = [min(y_test.min(), mdata["y_pred_test"].min()),
                    max(y_test.max(), mdata["y_pred_test"].max())]
            ax.plot(lims, lims, 'r--', lw=1.5)
            ax.set_xlabel("Gercek")
            ax.set_ylabel("Tahmin")
            ax.set_title(f"{mname}\nR2={mdata['test_r2']:.3f}", fontsize=8)
        fig_pred.suptitle(f"{target_col} - Gercek vs Tahmin", y=1.02)
        fig_pred.tight_layout()
        buf_pred = fig_to_buf(fig_pred)
        
        pdf.new_page()
        pdf.subtitle(f"{target_col} - Gercek vs Tahmin")
        pdf.add_image(buf_pred, width=520, height=260)
        
        # Residual grafiği
        fig_res, axes_res = plt.subplots(1, n_models, figsize=(4*n_models, 3.5))
        if n_models == 1:
            axes_res = [axes_res]
        for k, (mname, mdata) in enumerate(results.items()):
            ax = axes_res[k]
            residuals = y_test - mdata["y_pred_test"]
            ax.scatter(mdata["y_pred_test"], residuals, alpha=0.7, edgecolors='k', s=40)
            ax.axhline(y=0, color='r', linestyle='--', lw=1.5)
            ax.set_xlabel("Tahmin")
            ax.set_ylabel("Kalinti")
            ax.set_title(mname, fontsize=8)
        fig_res.suptitle(f"{target_col} - Residual Analizi", y=1.02)
        fig_res.tight_layout()
        buf_res = fig_to_buf(fig_res)
        
        pdf.new_page()
        pdf.subtitle(f"{target_col} - Residual (Kalinti) Analizi")
        pdf.add_image(buf_res, width=520, height=260)
        
        # R² Bar Chart
        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        model_names = list(results.keys())
        train_r2s = [results[m]["train_r2"] for m in model_names]
        test_r2s = [results[m]["test_r2"] for m in model_names]
        x_pos = np.arange(len(model_names))
        ax_bar.bar(x_pos - 0.2, train_r2s, 0.35, label='Train R2', color='steelblue')
        ax_bar.bar(x_pos + 0.2, test_r2s, 0.35, label='Test R2', color='coral')
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(model_names, rotation=25, ha='right', fontsize=8)
        ax_bar.set_ylabel("R2")
        ax_bar.set_title(f"{target_col} - Train vs Test R2")
        ax_bar.legend()
        ax_bar.set_ylim(0, 1.1)
        fig_bar.tight_layout()
        buf_bar = fig_to_buf(fig_bar)
        
        pdf.new_page()
        pdf.subtitle(f"{target_col} - R2 Karsilastirma")
        pdf.add_image(buf_bar, width=480, height=300)
        
        print(f"   ✅ {target_col} grafikleri oluşturuldu.")

    # ============ BOLUM 3: GUVENILIRLIK DENETIMI ============
    print("\n" + "="*60)
    print("  BOLUM 3: GUVENILIRLIK DENETIMI")
    print("="*60)
    
    pdf.new_page()
    pdf.title("3. Guvenilirlik Denetimi (Overfitting)")
    
    for target_col, results in all_results.items():
        pdf.subtitle(f"Hedef: {target_col}")
        for mname, mdata in results.items():
            diff = mdata["train_r2"] - mdata["test_r2"]
            if mdata["train_r2"] > 0.995 and diff > 0.15:
                status = "ASIRI OGRENME"
                print(f"   ❌ {target_col}/{mname}: ASIRI OGRENME (fark={diff:.4f})")
            elif diff > 0.10:
                status = "RISK"
                print(f"   ⚡ {target_col}/{mname}: RISK (fark={diff:.4f})")
            else:
                status = "GUVENLI"
                print(f"   ✅ {target_col}/{mname}: GUVENLI (fark={diff:.4f})")
            pdf.text(f"{mname}: Train R2={mdata['train_r2']:.4f}, Test R2={mdata['test_r2']:.4f}, Fark={diff:.4f} -> {status}")
        pdf.y -= 10

    # ============ BOLUM 4: COK AMACLI OPTIMIZASYON ============
    print("\n" + "="*60)
    print("  BOLUM 4: COK AMACLI OPTIMIZASYON (NSGA-II)")
    print("="*60)
    
    bounds = {}
    for inp in input_cols:
        n_unique = df[inp].nunique()
        if n_unique <= 5:
            bounds[inp] = {"type": "cat", "vals": sorted(df[inp].unique().tolist())}
        else:
            bounds[inp] = {"type": "float", "min": float(df[inp].min()), "max": float(df[inp].max())}
    
    def objective(trial):
        x_dict = {}
        for inp in input_cols:
            if bounds[inp]["type"] == "cat":
                x_dict[inp] = trial.suggest_categorical(inp, bounds[inp]["vals"])
            else:
                x_dict[inp] = trial.suggest_float(inp, bounds[inp]["min"], bounds[inp]["max"])
        
        x_arr = pd.DataFrame([x_dict])[input_cols].values
        x_scaled = master_scaler.transform(x_arr)
        
        preds = []
        for t in target_cols:
            preds.append(float(best_models[t].predict(x_scaled)[0]))
        return tuple(preds)
    
    print(f"   NSGA-II çalışıyor ({NSGA_TRIALS} deneme)...")
    study = optuna.create_study(directions=["minimize"] * len(target_cols))
    study.optimize(objective, n_trials=NSGA_TRIALS)
    
    pareto_data = []
    for pt in study.best_trials:
        row = pt.params.copy()
        for i, t in enumerate(target_cols):
            row[t] = pt.values[i]
        pareto_data.append(row)
    
    df_pareto = pd.DataFrame(pareto_data).drop_duplicates()
    print(f"   ✅ {len(df_pareto)} Pareto-optimal çözüm bulundu.")
    
    # Pareto Scatter (ilk 2 hedef üzerinden)
    if len(target_cols) >= 2:
        fig_par, ax_par = plt.subplots(figsize=(7, 5))
        ax_par.scatter(df_pareto[target_cols[0]], df_pareto[target_cols[1]], 
                       alpha=0.7, edgecolors='k', s=50, c='dodgerblue')
        ax_par.set_xlabel(target_cols[0])
        ax_par.set_ylabel(target_cols[1])
        ax_par.set_title("Pareto Siniri (NSGA-II)")
        ax_par.grid(True, alpha=0.3)
        fig_par.tight_layout()
        buf_par = fig_to_buf(fig_par)
        
        pdf.new_page()
        pdf.title("4. Cok Amacli Optimizasyon")
        pdf.subtitle("Pareto Siniri")
        pdf.add_image(buf_par, width=450, height=320)
        print("   ✅ Pareto grafiği oluşturuldu.")

    # ============ BOLUM 5: CRITIC + CODAS ============
    print("\n" + "="*60)
    print("  BOLUM 5: CRITIC AGIRLIKLARI & CODAS SIRALAMASI")
    print("="*60)
    
    df_targets_only = df_pareto[target_cols]
    weights = calculate_critic_weights(df_targets_only)
    
    print("\n   CRITIC Ağırlıkları:")
    for col, w in weights.items():
        print(f"      {col}: %{w*100:.2f}")
    
    scores = apply_codas(df_targets_only, weights)
    df_pareto["CODAS_Skoru"] = scores
    df_pareto = df_pareto.sort_values(by="CODAS_Skoru", ascending=False).reset_index(drop=True)
    df_pareto.index = df_pareto.index + 1
    
    print("\n   En İyi 5 Optimal Çözüm (CODAS):")
    top5 = df_pareto.head(5)
    print(top5.to_string())
    
    # PDF: CRITIC + CODAS
    pdf.new_page()
    pdf.title("5. CRITIC Agirliklari & CODAS Siralamasi")
    
    pdf.subtitle("CRITIC Agirliklari")
    for col, w in weights.items():
        pdf.text(f"{col}: %{w*100:.2f}")
    pdf.y -= 10
    
    pdf.subtitle("En Iyi 5 Optimal Cozum (CODAS)")
    all_cols_pareto = input_cols + target_cols + ["CODAS_Skoru"]
    headers_p = [c[:12] for c in all_cols_pareto]
    rows_p = []
    for _, row in top5.iterrows():
        rows_p.append([f"{row.get(c, ''):.4f}" if isinstance(row.get(c, ''), (int, float)) else str(row.get(c, '')) for c in all_cols_pareto])
    pdf.add_table(headers_p, rows_p)
    
    # Tüm Pareto setini de ekle
    pdf.new_page()
    pdf.subtitle(f"Tum Pareto Cozumleri ({len(df_pareto)} adet)")
    rows_all = []
    for _, row in df_pareto.iterrows():
        rows_all.append([f"{row.get(c, ''):.4f}" if isinstance(row.get(c, ''), (int, float)) else str(row.get(c, '')) for c in all_cols_pareto])
    pdf.add_table(headers_p, rows_all[:30])  # İlk 30'unu göster

    # ============ KAYDET ============
    pdf.save()
    
    # Pareto tablosunu da Excel'e kaydet
    excel_out = OUTPUT_PDF.replace(".pdf", "_pareto.xlsx")
    df_pareto.to_excel(excel_out, index=True)
    print(f"📊 Pareto tablosu kaydedildi: {os.path.abspath(excel_out)}")
    
    print("\n" + "="*60)
    print("  ✅ TÜM ANALİZ TAMAMLANDI!")
    print("="*60)


if __name__ == "__main__":
    run()
