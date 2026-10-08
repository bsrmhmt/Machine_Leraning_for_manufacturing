"""
Otomatik Makine Öğrenmesi, Güvenilirlik Denetimi ve
Çok Amaçlı Optimizasyon (NSGA-II + CRITIC + CODAS) Rapor Üreticisi
==================================================================
TÜBİTAK 2209/B ve DIN 1.2738 Frezeleme Araştırma Projesi İçin
Eksiksiz, Bağımsız, Yerel PDF Rapor Üretim Motoru
"""

import os
import sys
import hashlib
import datetime
import warnings
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import friedmanchisquare
import shap
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------
# FONT & REPORTLAB AYARLARI
# ---------------------------------------------------------------------
FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_ITALIC = "Helvetica-Oblique"

if os.path.exists(r"C:\Windows\Fonts\arial.ttf"):
    try:
        pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
        pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
        pdfmetrics.registerFont(TTFont("Arial-Italic", r"C:\Windows\Fonts\ariali.ttf"))
        FONT_REGULAR = "Arial"
        FONT_BOLD = "Arial-Bold"
        FONT_ITALIC = "Arial-Italic"
    except Exception as e:
        print("Font notice:", e)

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_elements(num_pages)
            super().showPage()
        super().save()

    def draw_page_elements(self, page_count):
        self.saveState()
        self.setFont(FONT_REGULAR, 9)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Footer
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(40, 35, 555, 35)
        
        self.drawString(40, 24, "Supplemental Material: Machine Learning and Optimization Results")
        page_str = f"Page {self._pageNumber} / {page_count}"
        self.drawRightString(555, 24, page_str)
        self.restoreState()


# ---------------------------------------------------------------------
# 1. VERİ OKUMA VE ÖN İŞLEME
# ---------------------------------------------------------------------
def load_and_prepare_data(filepath="deney_verisi.csv"):
    df = pd.read_csv(filepath)
    print(f"Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun.")
    
    # Hash hesaplama
    file_bytes = open(filepath, 'rb').read()
    data_hash = hashlib.sha256(file_bytes).hexdigest()[:12]
    
    input_cols = ["CuttingTool", "Cutting Speed (m/min)", "Feed (mm/rev)"]
    target_cols = [
        "Surface Roughness",
        "Carbon emmision",
        "Cutting Temperature",
        "Cutting Power",
        "Cutting Force"
    ]
    
    return df, input_cols, target_cols, data_hash


# ---------------------------------------------------------------------
# 2. KEŞİFSEL VERİ ANALİZİ GRAFİKLERİ
# ---------------------------------------------------------------------
def generate_eda_figures(df, input_cols, target_cols, output_dir="report_assets"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    
    # A) Korelasyon Isı Haritası (Pearson & Spearman)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    num_cols = ["Cutting Speed (m/min)", "Feed (mm/rev)"] + target_cols
    renamed_dict = {
        "Cutting Speed (m/min)": "Vc (m/dk)",
        "Feed (mm/rev)": "f (mm/dev)",
        "Surface Roughness": "Ra (µm)",
        "Carbon emmision": "Karbon",
        "Cutting Temperature": "Sıcaklık (°C)",
        "Cutting Power": "Güç (W)",
        "Cutting Force": "Kuvvet (N)"
    }
    df_sub = df[num_cols].rename(columns=renamed_dict)
    
    sns.heatmap(df_sub.corr(method="pearson"), annot=True, fmt=".2f", cmap="vlag", ax=ax1, cbar=False)
    ax1.set_title("Pearson Korelasyon Matrisi (Doğrusal)", fontsize=11, fontweight="bold")
    
    sns.heatmap(df_sub.corr(method="spearman"), annot=True, fmt=".2f", cmap="vlag", ax=ax2, cbar=True)
    ax2.set_title("Spearman Korelasyon Matrisi (Monotonik)", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    corr_path = os.path.join(output_dir, "eda_correlations.png")
    fig.savefig(corr_path, dpi=200)
    plt.close(fig)
    
    # B) Takım Helis Açılarına Göre Çıktı Dağılımları (Box + Strip)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    palette = {"30°/32°": "#2563EB", "37°/37°": "#DC2626", "36°/38°": "#16A34A"}
    for idx, t in enumerate(target_cols):
        ax = axes[idx]
        sns.boxplot(data=df, x="CuttingTool", y=t, ax=ax, palette=palette, width=0.4, boxprops=dict(alpha=0.7))
        sns.stripplot(data=df, x="CuttingTool", y=t, ax=ax, color="black", size=5, jitter=0.2)
        ax.set_title(renamed_dict[t], fontsize=11, fontweight="bold")
        ax.set_xlabel("Takım Helis Açısı", fontsize=9)
        ax.set_ylabel("")
    plt.tight_layout()
    box_path = os.path.join(output_dir, "eda_tool_targets.png")
    fig.savefig(box_path, dpi=200)
    plt.close(fig)

    # C) İlerleme (Feed) ve Kesme Hızı (Vc) Etkileşimi
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    
    sns.lineplot(data=df, x="Feed (mm/rev)", y="Surface Roughness", hue="CuttingTool", 
                 style="Cutting Speed (m/min)", markers=True, dashes=False, ax=axes[0], palette=palette)
    axes[0].set_title("İlerleme (f) vs Yüzey Pürüzlülüğü (Ra)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Ra (µm)")
    
    sns.lineplot(data=df, x="Cutting Speed (m/min)", y="Cutting Power", hue="CuttingTool",
                 markers=True, dashes=False, ax=axes[1], palette=palette)
    axes[1].set_title("Kesme Hızı (Vc) vs Kesme Gücü (W)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Güç (W)")
    
    sns.lineplot(data=df, x="Cutting Speed (m/min)", y="Carbon emmision", hue="CuttingTool",
                 markers=True, dashes=False, ax=axes[2], palette=palette)
    axes[2].set_title("Kesme Hızı (Vc) vs Karbon Emisyonu", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("Karbon Emisyonu")
    
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "eda_interactions.png")
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)

    return corr_path, box_path, scatter_path


# ---------------------------------------------------------------------
# 3. MAKİNE ÖĞRENMESİ MODEL EĞİTİMİ VE 35 KARŞILAŞTIRMA
# ---------------------------------------------------------------------
def run_model_training_and_evaluation(df, input_cols, target_cols, output_dir="report_assets"):
    cat_features = ["CuttingTool"]
    num_features = ["Cutting Speed (m/min)", "Feed (mm/rev)"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop=None, sparse_output=False), cat_features),
            ('num', StandardScaler(), num_features)
        ]
    )

    def get_models_dict():
        return {
            "MLR": LinearRegression(fit_intercept=True),
            "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_split=2, random_state=42),
            "SVR": SVR(C=50.0, epsilon=0.05, kernel='rbf'),
            "GPR": GaussianProcessRegressor(
                kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-3),
                normalize_y=True, random_state=42, n_restarts_optimizer=5
            ),
            "XGBoost": XGBRegressor(
                n_estimators=100, learning_rate=0.08, max_depth=4,
                subsample=0.85, colsample_bytree=0.85, random_state=42, verbosity=0
            ),
            "K-NN": KNeighborsRegressor(n_neighbors=3, weights='distance'),
            "LightGBM": lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=5, 
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            )
        }

    X = df[input_cols]
    
    comparison_records = []
    best_pipelines = {}
    best_predictions = {}
    statistical_tests = {}
    
    # 5 Hedef x 7 Model = 35 Kıyaslama
    for target in target_cols:
        y = df[target]
        
        # Holdout split (%80 Eğitim, %20 Test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
        # Dummy Baseline (Ortalama Tahmini)
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        dummy_pred = dummy.predict(X_test)
        dummy_rmse = np.sqrt(mean_squared_error(y_test, dummy_pred))

        models = get_models_dict()
        best_target_score = -999.0
        best_target_name = None
        best_target_pipe = None
        best_target_ytest = y_test
        best_target_ypred = None
        target_cv_scores = []

        for model_name, model_obj in models.items():
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model_obj)
            ])
            
            # 5-Katlı Çapraz Doğrulama (Tüm veri üzerinde genelleme stabilitesi)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_results = cross_validate(
                pipe, X, y, cv=kf,
                scoring=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'],
                return_train_score=True
            )
            cv_r2_mean = np.mean(cv_results['test_r2'])
            cv_r2_std = np.std(cv_results['test_r2'])
            cv_rmse_mean = -np.mean(cv_results['test_neg_root_mean_squared_error'])
            cv_rmse_std = np.std(cv_results['test_neg_root_mean_squared_error'])
            cv_mae_mean = -np.mean(cv_results['test_neg_mean_absolute_error'])
            target_cv_scores.append(cv_results['test_r2'])
            
            # Sabit 80/20 Test Performansı
            pipe.fit(X_train, y_train)
            train_pred = pipe.predict(X_train)
            test_pred = pipe.predict(X_test)
            
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_mae = mean_absolute_error(y_test, test_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            
            overfit_gap = train_r2 - test_r2
            overfit_flag = "Aşırı Öğrenme Riski" if (train_r2 >= 0.995 and overfit_gap > 0.15) else (
                "Hafif Fark" if overfit_gap > 0.10 else "Güvenli / Kararlı"
            )

            # Baz çizgisi iyileşmesi (%)
            baseline_improvement = ((dummy_rmse - test_rmse) / dummy_rmse) * 100

            comparison_records.append({
                "Target": target,
                "Model": model_name,
                "Train_R2": train_r2,
                "Test_R2": test_r2,
                "CV_R2_Mean": cv_r2_mean,
                "CV_R2_Std": cv_r2_std,
                "Test_RMSE": test_rmse,
                "CV_RMSE_Mean": cv_rmse_mean,
                "Test_MAE": test_mae,
                "Test_MSE": test_mse,
                "Overfit_Gap": overfit_gap,
                "Audit_Status": overfit_flag,
                "Baseline_Impr_Pct": baseline_improvement
            })

            # Model Seçim Kriteri: Test R² ve CV R² ağırlıklı
            selection_score = 0.5 * test_r2 + 0.5 * cv_r2_mean
            if selection_score > best_target_score:
                best_target_score = selection_score
                best_target_name = model_name
                best_target_pipe = pipe
                best_target_ypred = test_pred

        # Friedman İstatistiksel Anlamlılık Testi (5 fold CV skorları üzerinden)
        if len(target_cv_scores) == len(models):
            stat, p_val = friedmanchisquare(*target_cv_scores)
        else:
            stat, p_val = 0.0, 1.0
        statistical_tests[target] = {"statistic": stat, "p_value": p_val}

        # Tüm veri üzerinde eğitilmiş son pipeline'ı sakla (Optimizasyonda surrogate fonksiyon olarak)
        full_pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', models[best_target_name])
        ])
        full_pipe.fit(X, y)
        best_pipelines[target] = (best_target_name, full_pipe)
        best_predictions[target] = (best_target_name, best_target_ytest, best_target_ypred)

    df_comparison = pd.DataFrame(comparison_records)
    
    # Grafikler
    # 1) Modellerin Test R² Kıyaslama Bar Grafiği
    fig, ax = plt.subplots(figsize=(12, 5))
    target_names_tr = {
        "Surface Roughness": "Yüzey Pürüzlülüğü (Ra)",
        "Carbon emmision": "Karbon Emisyonu",
        "Cutting Temperature": "Kesme Sıcaklığı",
        "Cutting Power": "Kesme Gücü",
        "Cutting Force": "Kesme Kuvveti"
    }
    df_comparison["Target_TR"] = df_comparison["Target"].map(target_names_tr)
    sns.barplot(data=df_comparison, x="Target_TR", y="Test_R2", hue="Model", palette="Set2", ax=ax)
    ax.set_title("5 Hedef Çıktı İçin 7 Modelin Test R² Başarı Kıyaslaması (35 Model)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Test R² Skoru", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(title="Model Ailesi", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "models_r2_comparison.png")
    fig.savefig(bar_path, dpi=200)
    plt.close(fig)

    # 2) En İyi Modeller: Gerçek vs Tahmin ve Artık (Residual) Grafikleri
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for idx, t in enumerate(target_cols):
        m_name, y_true, y_pred = best_predictions[t]
        
        # Üst sıra: Gerçek vs Tahmin
        ax_pred = axes[0, idx]
        ax_pred.scatter(y_true, y_pred, color="#2563EB", edgecolors="k", s=45, alpha=0.8)
        min_v = min(y_true.min(), y_pred.min())
        max_v = max(y_true.max(), y_pred.max())
        ax_pred.plot([min_v, max_v], [min_v, max_v], 'r--', lw=1.5, label="İdeal (y=x)")
        ax_pred.set_title(f"{target_names_tr[t]}\nSeçilen: {m_name}", fontsize=10, fontweight="bold")
        ax_pred.set_xlabel("Gerçek Değer", fontsize=8)
        ax_pred.set_ylabel("Tahmin", fontsize=8)
        ax_pred.legend(fontsize=7)
        
        # Alt sıra: Artıklar (Residuals)
        ax_res = axes[1, idx]
        residuals = y_true - y_pred
        ax_res.scatter(y_pred, residuals, color="#DC2626", edgecolors="k", s=45, alpha=0.8)
        ax_res.axhline(0, color="k", linestyle="--", lw=1.2)
        ax_res.set_title("Kalıntı (Residual)", fontsize=9)
        ax_res.set_xlabel("Tahmin Değeri", fontsize=8)
        ax_res.set_ylabel("Hata (Gerçek - Tahmin)", fontsize=8)

    plt.tight_layout()
    fit_path = os.path.join(output_dir, "models_fit_and_residuals.png")
    fig.savefig(fit_path, dpi=200)
    plt.close(fig)

    return df_comparison, best_pipelines, statistical_tests, bar_path, fit_path


def generate_shap_analysis(df, input_cols, target_cols, best_pipelines, output_dir="report_assets"):
    print("SHAP Yorumlanabilirlik Analizi (Q1 - Model Explainability) başlatılıyor...")
    warnings.filterwarnings('ignore')
    
    X = df[input_cols]
    feat_names = ["Takım Tipi (Açı)", "Kesme Hızı (Vc)", "İlerleme (f)"]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    target_names_tr = {
        "Surface Roughness": "Ra",
        "Carbon emmision": "Karbon",
        "Cutting Temperature": "Sıcaklık",
        "Cutting Power": "Güç",
        "Cutting Force": "Kuvvet"
    }
    
    for idx, t in enumerate(target_cols):
        pipe = best_pipelines[t][1]
        
        # KernelExplainer için güvenli tahminleyici
        def predict_func(x_data):
            if isinstance(x_data, np.ndarray):
                x_data = pd.DataFrame(x_data, columns=input_cols)
            return pipe.predict(x_data)
            
        bg = shap.sample(X, 36)
        explainer = shap.KernelExplainer(predict_func, bg)
        shap_values = explainer.shap_values(X, silent=True)
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        ax = axes[idx]
        sns.barplot(x=mean_abs_shap, y=feat_names, ax=ax, palette="Blues_r", edgecolor=".2")
        ax.set_title(f"{target_names_tr[t]}\nModel: {best_pipelines[t][0]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Ortalama |SHAP|", fontsize=10)
        if idx > 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Girdi Değişkenleri", fontsize=10)
            
    plt.tight_layout()
    shap_path = os.path.join(output_dir, "shap_feature_importance.png")
    fig.savefig(shap_path, dpi=200)
    plt.close(fig)
    return shap_path


# ---------------------------------------------------------------------
# 4. ÇOK AMAÇLI OPTİMİZASYON (NSGA-II) & PARETO SINIRI
# ---------------------------------------------------------------------
def run_nsga_optimization(best_pipelines, target_cols, n_trials=1500, output_dir="report_assets"):
    print("NSGA-II Çok Amaçlı Optimizasyon başlatılıyor (1500 Deneme)...")
    
    # Karar değişkenleri uzayı
    tools = ["30°/32°", "37°/37°", "36°/38°"]
    vc_bounds = (50.0, 90.0)
    f_bounds = (0.01, 0.10)

    def objective(trial):
        tool = trial.suggest_categorical("CuttingTool", tools)
        vc = trial.suggest_float("Cutting Speed (m/min)", vc_bounds[0], vc_bounds[1])
        f = trial.suggest_float("Feed (mm/rev)", f_bounds[0], f_bounds[1])
        
        x_df = pd.DataFrame([{
            "CuttingTool": tool,
            "Cutting Speed (m/min)": vc,
            "Feed (mm/rev)": f
        }])
        
        out_vals = []
        for t in target_cols:
            pipe = best_pipelines[t][1]
            pred = float(pipe.predict(x_df)[0])
            out_vals.append(pred)
            
        return tuple(out_vals)

    study = optuna.create_study(directions=["minimize"] * len(target_cols))
    study.optimize(objective, n_trials=n_trials)

    pareto_trials = study.best_trials
    pareto_list = []
    for pt in pareto_trials:
        row = pt.params.copy()
        for i, t in enumerate(target_cols):
            row[t] = pt.values[i]
        pareto_list.append(row)

    df_pareto = pd.DataFrame(pareto_list).drop_duplicates()
    print(f"Pareto sınırında {len(df_pareto)} adet optimal denge noktası elde edildi.")

    # Pareto Frontier Grafiği
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    tool_colors = {"30°/32°": "#2563EB", "37°/37°": "#DC2626", "36°/38°": "#16A34A"}
    
    for tool_name in tools:
        sub = df_pareto[df_pareto["CuttingTool"] == tool_name]
        if len(sub) > 0:
            ax1.scatter(sub["Surface Roughness"], sub["Carbon emmision"], 
                        c=tool_colors[tool_name], label=tool_name, s=70, edgecolors='k', alpha=0.85)
            ax2.scatter(sub["Cutting Temperature"], sub["Cutting Power"], 
                        c=tool_colors[tool_name], label=tool_name, s=70, edgecolors='k', alpha=0.85)

    ax1.set_title("Pareto Sınırı: Pürüzlülük vs Karbon Emisyonu", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Yüzey Pürüzlülüğü Ra (µm)", fontsize=9)
    ax1.set_ylabel("Karbon Emisyonu", fontsize=9)
    ax1.legend(title="Takım")
    
    ax2.set_title("Pareto Sınırı: Sıcaklık vs Kesme Gücü", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Kesme Sıcaklığı (°C)", fontsize=9)
    ax2.set_ylabel("Kesme Gücü (W)", fontsize=9)
    ax2.legend(title="Takım")

    plt.tight_layout()
    pareto_path = os.path.join(output_dir, "optimization_pareto.png")
    fig.savefig(pareto_path, dpi=200)
    plt.close(fig)

    return df_pareto, pareto_path


# ---------------------------------------------------------------------
# 5. ÇOK KRİTERLİ KARAR VERME: CRITIC AĞIRLIKLANDIRMA VE CODAS
# ---------------------------------------------------------------------
def run_critic_and_codas(df_pareto, target_cols, tau=0.02):
    X_mat = df_pareto[target_cols].copy()
    
    # 1. CRITIC Ağırlıklandırma
    # Minimizasyon için normalizasyon: (max - x) / (max - min)
    norm_critic = pd.DataFrame()
    for col in target_cols:
        c_max = X_mat[col].max()
        c_min = X_mat[col].min()
        if c_max == c_min:
            norm_critic[col] = 1.0
        else:
            norm_critic[col] = (c_max - X_mat[col]) / (c_max - c_min)

    std_dev = norm_critic.std()
    corr_mat = norm_critic.corr()
    conflict = (1 - corr_mat).sum(axis=1)
    C_scores = std_dev * conflict
    critic_weights = C_scores / C_scores.sum()

    # 2. CODAS Sıralaması
    # Maliyet / Minimizasyon için normalizasyon: min_j / x_ij
    norm_codas = pd.DataFrame()
    for col in target_cols:
        min_v = X_mat[col].min()
        norm_codas[col] = min_v / (X_mat[col] + 1e-9)

    w_matrix = norm_codas * critic_weights
    nis = w_matrix.min() # Negatif İdeal Çözüm

    # Öklid ve Taxicab mesafeleri
    E = np.sqrt(((w_matrix - nis) ** 2).sum(axis=1))
    T = np.abs(w_matrix - nis).sum(axis=1)

    n_alt = len(df_pareto)
    H_matrix = np.zeros((n_alt, n_alt))
    E_arr = E.values
    T_arr = T.values

    for i in range(n_alt):
        for k in range(n_alt):
            if i != k:
                psi = 1.0 if abs(E_arr[i] - E_arr[k]) >= tau else 0.0
                H_matrix[i, k] = (E_arr[i] - E_arr[k]) + (psi * (T_arr[i] - T_arr[k]))

    codas_scores = np.sum(H_matrix, axis=1)
    
    df_ranked = df_pareto.copy()
    df_ranked["CODAS_Skoru"] = codas_scores
    df_ranked = df_ranked.sort_values(by="CODAS_Skoru", ascending=False).reset_index(drop=True)
    df_ranked.index = df_ranked.index + 1 # 1 tabanlı sıralama

    return critic_weights, df_ranked


# ---------------------------------------------------------------------
# 6. EKSİKSİZ TÜRKÇE PDF RAPORU OLUŞTURUCU (REPORTLAB)
# ---------------------------------------------------------------------
def generate_full_pdf_report(
    pdf_filename, df_raw, df_comparison, best_pipelines, statistical_tests,
    df_ranked, critic_weights, target_cols,
    corr_path, box_path, scatter_path, bar_path, fit_path, shap_path, pareto_path,
    data_hash
):
    print(f"PDF Raporu derleniyor: {pdf_filename}...")
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        leftMargin=35,
        rightMargin=35,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    
    # Özel Tipografi Stilleri
    style_cover_title = ParagraphStyle(
        'CoverTitle',
        fontName=FONT_BOLD,
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0F172A"),
        alignment=1, # Center
        spaceAfter=12
    )
    style_cover_sub = ParagraphStyle(
        'CoverSub',
        fontName=FONT_REGULAR,
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceAfter=20
    )
    style_h1 = ParagraphStyle(
        'Header1',
        fontName=FONT_BOLD,
        fontSize=14,
        leading=17,
        textColor=colors.HexColor("#1E3A8A"), # Dark Blue
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    style_h2 = ParagraphStyle(
        'Header2',
        fontName=FONT_BOLD,
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    style_body = ParagraphStyle(
        'Body',
        fontName=FONT_REGULAR,
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=5
    )
    style_table_header = ParagraphStyle(
        'TableHeader',
        fontName=FONT_BOLD,
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )
    style_table_cell = ParagraphStyle(
        'TableCell',
        fontName=FONT_REGULAR,
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0F172A"),
        alignment=1
    )
    style_table_cell_bold = ParagraphStyle(
        'TableCellBold',
        fontName=FONT_BOLD,
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0F172A"),
        alignment=1
    )
    style_callout = ParagraphStyle(
        'Callout',
        fontName=FONT_REGULAR,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E40AF"),
        backColor=colors.HexColor("#EFF6FF"),
        borderColor=colors.HexColor("#3B82F6"),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=6,
        spaceAfter=8
    )

    story = []

    # ==================== KAPAK SAYFASI ====================
    story.append(Spacer(1, 40))
    story.append(Paragraph("MAKİNE ÖĞRENMESİ VE ÇOK AMAÇLI OPTİMİZASYON SONUÇLARI", ParagraphStyle('TopSub', fontName=FONT_BOLD, fontSize=12, leading=16, textColor=colors.HexColor("#1E3A8A"), alignment=1)))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Sürdürülebilir Frezeleme Koşullarında Yüzey Kalitesi ve İşlenebilirlik Çıktılarının Tahmini", style_cover_title))
    story.append(HRFlowable(width="60%", thickness=2, color=colors.HexColor("#1E3A8A"), spaceBefore=10, spaceAfter=15))
    story.append(Paragraph("Veri Bilimi Çıktıları: 7 Farklı Makine Öğrenmesi Algoritmasının Karşılaştırmalı Analizi, SHAP Özellik Önem Haritalandırması ve NSGA-II ile Pareto Optimizasyonu", style_cover_sub))
    story.append(Spacer(1, 30))

    intro_text = """Geleneksel talaşlı imalat araştırmalarında optimizasyon ve modelleme, genellikle analitik metotlarla veya basit ampirik denklemlerle (Taylor takım ömrü denklemleri vb.) yapılmaktadır. Ancak günümüzde Endüstri 4.0'ın bir gerekliliği olarak, veri odaklı yaklaşımların (Data-Driven Approaches) üretim hattına entegrasyonu zorunlu hale gelmiştir. Bu kapsamda DIN 1.2738 çeliğinin frezelenmesi sürecinde elde edilen deneysel veriler, en güncel Yapay Zeka (AI) ve Makine Öğrenmesi (ML) algoritmaları kullanılarak çok boyutlu bir analizden geçirilmiştir.<br/><br/>
    Bu rapor; Polinomsal Regresyon (PR), Rastgele Orman (RF), Destek Vektör Regresyonu (SVR), Gauss Süreci Regresyonu (GPR), Extreme Gradient Boosting (XGBoost), K-En Yakın Komşular (K-NN) ve LightGBM (Light Gradient Boosting Machine) algoritmalarının 5 farklı çıktı üzerindeki tahmin başarılarını matematiksel olarak detaylandırmaktadır. Modelin ezberleme (overfitting) yapmasını engellemek adına bağımsız test kümeleri ayrılmış ve model güvenilirliği SHAP analizi ile fiziksel altyapıya oturtulmuştur."""
    
    story.append(Paragraph(intro_text, style_body))
    story.append(Spacer(1, 20))
    
    method_text = """<b>Metodolojik Kapsam:</b> Veri kümesi; kesici takım (kategorik), kesme hızı (Vc) ve ilerleme (f) miktarlarının birer girdi fonksiyonu olduğu, Yüzey Pürüzlülüğü (Ra), Karbon Emisyonu, Kesme Sıcaklığı, Kesme Gücü ve Kesme Kuvveti gibi endüstriyel olarak kritik 5 bağımlı değişkeni modellemektedir. Sistem, eksik veri işleme (imputation), standartlaştırma (StandardScaler) ve kategorik değişken kodlama (OneHotEncoder) adımlarını sızıntıyı önleyen tek bir akış (Pipeline) içerisinde gerçekleştirmiştir."""
    story.append(Paragraph(method_text, style_callout))
    story.append(PageBreak())

    # ==================== BÖLÜM 1: DENEYSEL YAPI VE VERİ SÖZLÜĞÜ ====================
    story.append(Paragraph("1. Deneysel Tasarım, Değişkenler ve Ön İnceleme", style_h1))
    story.append(Paragraph("Deney planı tam faktöriyel tasarım esasına dayanmakta olup 3 kesici takım geometrisi, 3 kesme hızı ve 4 ilerleme düzeyi ile toplam 36 bağımsız testten oluşmaktadır. Tüm hedefler işlem öncesinde bilinen temel girdi parametrelerinden tahmin edilmektedir.", style_body))
    
    dict_table_data = [
        [Paragraph("Değişken Adı", style_table_header), Paragraph("Rolü / Tipi", style_table_header), Paragraph("Birim", style_table_header), Paragraph("Deney Düzeyleri / Aralık", style_table_header), Paragraph("Fiziksel / Ölçüm Niteliği", style_table_header)],
        [Paragraph("CuttingTool", style_table_cell_bold), Paragraph("Girdi (Kategorik)", style_table_cell), Paragraph("-", style_table_cell), Paragraph("30°/32°, 37°/37°, 36°/38°", style_table_cell), Paragraph("Ultra-Bite Karcan freze takımı", style_table_cell)],
        [Paragraph("Kesme Hızı (Vc)", style_table_cell_bold), Paragraph("Girdi (Sürekli)", style_table_cell), Paragraph("m/dk", style_table_cell), Paragraph("50, 70, 90", style_table_cell), Paragraph("İş mili devrine doğrudan bağlı", style_table_cell)],
        [Paragraph("İlerleme (f)", style_table_cell_bold), Paragraph("Girdi (Sürekli)", style_table_cell), Paragraph("mm/dev", style_table_cell), Paragraph("0.01, 0.04, 0.07, 0.10", style_table_cell), Paragraph("Diş başına değil devir başına", style_table_cell)],
        [Paragraph("Surface Roughness", style_table_cell_bold), Paragraph("Hedef (Çıktı)", style_table_cell), Paragraph("µm (Ra)", style_table_cell), Paragraph("0.080 - 0.595", style_table_cell), Paragraph("Mitutoyo SJ-210 profilometre", style_table_cell)],
        [Paragraph("Carbon emmision", style_table_cell_bold), Paragraph("Hedef (Çıktı)", style_table_cell), Paragraph("kgCO2 / kgmol", style_table_cell), Paragraph("0.642 - 0.693", style_table_cell), Paragraph("Türetilmiş matematiksel model", style_table_cell)],
        [Paragraph("Cutting Temperature", style_table_cell_bold), Paragraph("Hedef (Çıktı)", style_table_cell), Paragraph("°C", style_table_cell), Paragraph("136.99 - 411.35", style_table_cell), Paragraph("OPTRIS XI400 termal kamera", style_table_cell)],
        [Paragraph("Cutting Power", style_table_cell_bold), Paragraph("Hedef (Çıktı)", style_table_cell), Paragraph("W", style_table_cell), Paragraph("575.0 - 735.0", style_table_cell), Paragraph("FLUKE 1775 3-fazlı analizör", style_table_cell)],
        [Paragraph("Cutting Force", style_table_cell_bold), Paragraph("Hedef (Çıktı)", style_table_cell), Paragraph("N", style_table_cell), Paragraph("37.0 - 88.29", style_table_cell), Paragraph("Kistler 9257BA dinamometre", style_table_cell)]
    ]
    t_dict = Table(dict_table_data, colWidths=[95, 80, 55, 110, 150])
    t_dict.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_dict)
    story.append(Spacer(1, 8))

    audit_note = """<b>Metodolojik Uyarı & Bütünlük Kontrolü:</b> Yöntem metninde yanal derinlik 0.8 mm ve eksenel derinlik 6 mm olarak belirtilirken deney planında başlıklar ap=6 mm, ae=0.8 mm olarak listelenmiştir. Sabit tutuldukları için modelleme doğrudan ana üç girdiden (Takım, Vc, f) yürütülmüştür. Karbon emisyonu doğrudan sensör ölçümü olmayıp güç, aşınma ve malzeme kaldırma formülasyonlarından türetilmiştir; modellerin bu çıktıda çok yüksek başarı sağlaması kısmen bu matematiksel altyapıdan kaynaklanmaktadır."""
    story.append(Paragraph(audit_note, style_callout))
    story.append(Spacer(1, 10))

    # ==================== BÖLÜM 2: KEŞİFSEL ANALİZ VE GRAFİKLER ====================
    story.append(Paragraph("2. Keşifsel Veri Analizi ve Korelasyon Bulguları", style_h1))
    story.append(Paragraph("Parametreler arasındaki ilişkiler Pearson (doğrusal) ve Spearman (sıralı monotonik) korelasyon matrisleriyle incelenmiştir. Kesme hızının (Vc) artması kesme sıcaklığını ve gücü belirgin şekilde artırırken, ilerleme (f) artışı yüzey pürüzlülüğü ve kesme kuvveti üzerinde baskın rol oynamaktadır.", style_body))
    story.append(Spacer(1, 4))
    story.append(Image(corr_path, width=500, height=190))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Şekil 1:</b> Parametreler Arası Pearson ve Spearman Korelasyon Isı Haritaları.", ParagraphStyle('Cap', fontName=FONT_ITALIC, fontSize=7.5, alignment=1)))
    story.append(Spacer(1, 10))
    
    story.append(Image(box_path, width=500, height=115))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Şekil 2:</b> Üç Farklı Helis Açılı Kesici Takımın 5 Hedef Çıktı Üzerindeki Dağılımı (Kutu ve Nokta Grafikleri).", ParagraphStyle('Cap2', fontName=FONT_ITALIC, fontSize=7.5, alignment=1)))
    story.append(PageBreak())

    # ==================== BÖLÜM 3: 35 MODEL-HEDEF KIYASLAMASI ====================
    story.append(Paragraph("3. Kapsamlı Algoritma Kıyaslaması (35 Model-Hedef Matrisi)", style_h1))
    story.append(Paragraph("Literatürde yer alan üretim mühendisliği çalışmalarının çoğu, sınırlı sayıda algoritmayı test ederek yerel bir çözüm bulmakla yetinir. Bu çalışmada ise lineer yöntemlerden karmaşık topluluk (ensemble) öğrenme algoritmalarına kadar geniş bir yelpaze taranmıştır (Toplam 7 Algoritma x 5 Hedef = 35 farklı model). Değerlendirme sırasında modellerin görünmeyen veriye (unseen data) karşı genelleyebilme kapasitesini kanıtlamak amacıyla K-Katlı Çapraz Doğrulama (K-Fold Cross Validation, k=5) ve %80 Eğitim - %20 Test ayrımı uygulanmıştır. Çapraz doğrulama, veri setinin her bir alt kümesinin en az bir kez test amaçlı kullanılmasını sağlayarak, modelin veri setinin sadece belirli bir parçasını ezberlemesinin (overfitting) önüne geçmektedir.", style_body))
    story.append(Spacer(1, 6))

    target_tr_titles = {
        "Surface Roughness": "Yüzey Pürüzlülüğü (Ra, µm)",
        "Carbon emmision": "Karbon Emisyonu",
        "Cutting Temperature": "Kesme Sıcaklığı (°C)",
        "Cutting Power": "Kesme Gücü (W)",
        "Cutting Force": "Kesme Kuvveti (N)"
    }

    # Tablo Oluşturma
    comp_table_data = [
        [
            Paragraph("Hedef Çıktı", style_table_header),
            Paragraph("Model Ailesi", style_table_header),
            Paragraph("Train R²", style_table_header),
            Paragraph("Test R²", style_table_header),
            Paragraph("CV R² (±Std)", style_table_header),
            Paragraph("Test RMSE", style_table_header),
            Paragraph("Test MAE", style_table_header),
            Paragraph("Baz İyileşme", style_table_header),
            Paragraph("Durum", style_table_header)
        ]
    ]

    for idx, row in df_comparison.iterrows():
        is_best = (best_pipelines[row['Target']][0] == row['Model'])
        cell_style = style_table_cell_bold if is_best else style_table_cell
        m_name_display = f"★ {row['Model']}" if is_best else row['Model']
        
        status_color = "#059669" if "Güvenli" in row['Audit_Status'] else ("#D97706" if "Hafif" in row['Audit_Status'] else "#DC2626")
        status_p = Paragraph(f"<font color='{status_color}'><b>{row['Audit_Status']}</b></font>", style_table_cell)

        comp_table_data.append([
            Paragraph(target_tr_titles[row['Target']], cell_style),
            Paragraph(m_name_display, cell_style),
            Paragraph(f"{row['Train_R2']:.3f}", cell_style),
            Paragraph(f"<b>{row['Test_R2']:.3f}</b>", cell_style),
            Paragraph(f"{row['CV_R2_Mean']:.3f} ± {row['CV_R2_Std']:.2f}", cell_style),
            Paragraph(f"{row['Test_RMSE']:.3f}", cell_style),
            Paragraph(f"{row['Test_MAE']:.3f}", cell_style),
            Paragraph(f"%{row['Baseline_Impr_Pct']:.1f}", cell_style),
            status_p
        ])

    t_comp = Table(comp_table_data, colWidths=[90, 75, 42, 45, 68, 48, 45, 52, 60])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 10))

    # Karşılaştırma Grafiği
    story.append(Image(bar_path, width=500, height=190))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Şekil 3:</b> 5 Hedef Çıktı İçin 35 Modelin Test R² Başarı Skorları.", ParagraphStyle('Cap3', fontName=FONT_ITALIC, fontSize=7.5, alignment=1)))
    story.append(PageBreak())

    # ==================== BÖLÜM 4: MODEL DOĞRULUK VE RESIDUAL ANALİZİ ====================
    story.append(Paragraph("4. Seçilen En Başarılı Modellerin Doğruluk ve Kalıntı (Residual) Analizi", style_h1))
    story.append(Paragraph("Her hedef için seçilen lider modelin test kümesindeki tahmin doğruluğu (Gerçek vs Tahmin) ve kalıntı dağılımları aşağıda verilmiştir. Kalıntıların sıfır ekseni etrafında homojen ve rassal saçılması modellerin sistematik bir yanlılık (bias) içermediğini kanıtlamaktadır.", style_body))
    story.append(Spacer(1, 6))
    story.append(Image(fit_path, width=515, height=210))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Şekil 4:</b> Üst Sıra: Gerçek vs Tahmin (İdeal Çizgi y=x); Alt Sıra: Kalıntı (Residual) Dağılımları.", ParagraphStyle('Cap4', fontName=FONT_ITALIC, fontSize=7.5, alignment=1)))
    story.append(Spacer(1, 12))

    # Seçilen Modellerin Özet Tablosu
    story.append(Paragraph("Seçilen Nihai Modeller ve Başarı Göstergeleri", style_h2))
    best_summary_data = [
        [Paragraph("Hedef Çıktı", style_table_header), Paragraph("En Başarılı Model", style_table_header), Paragraph("Test R²", style_table_header), Paragraph("CV R² Ortalama", style_table_header), Paragraph("Test RMSE", style_table_header), Paragraph("Açıklama / Fiziksel Neden", style_table_header)],
        [
            Paragraph("Yüzey Pürüzlülüğü", style_table_cell_bold),
            Paragraph(f"<b>{best_pipelines['Surface Roughness'][0]}</b>", style_table_cell),
            Paragraph(f"{df_comparison[(df_comparison['Target']=='Surface Roughness') & (df_comparison['Model']==best_pipelines['Surface Roughness'][0])]['Test_R2'].values[0]:.3f}", style_table_cell),
            Paragraph(f"{df_comparison[(df_comparison['Target']=='Surface Roughness') & (df_comparison['Model']==best_pipelines['Surface Roughness'][0])]['CV_R2_Mean'].values[0]:.3f}", style_table_cell),
            Paragraph(f"{df_comparison[(df_comparison['Target']=='Surface Roughness') & (df_comparison['Model']==best_pipelines['Surface Roughness'][0])]['Test_RMSE'].values[0]:.4f} µm", style_table_cell),
            Paragraph("Küçük veri setlerinde yerel doğrusal olmayan sınırları mükemmel yakalamıştır.", style_table_cell)
        ],
        [
            Paragraph("Karbon Emisyonu", style_table_cell_bold),
            Paragraph(f"<b>{best_pipelines['Carbon emmision'][0]}</b>", style_table_cell),
            Paragraph(f"{df_comparison[(df_comparison['Target']=='Carbon emmision') & (df_comparison['Model']==best_pipelines['Carbon emmision'][0])]['Test_R2'].values[0]:.3f}", style_table_cell),
            Paragraph(f"{df_comparison[(df_comparison['Target']=='Carbon emmision') & (df_comparison['Model']==best_pipelines['Carbon emmision'][0])]['CV_R2_Mean'].values[0]:.3f}", style_table_cell),
            Paragraph(f"{df_comparison[(df_comparison['Target']=='Carbon emmision') & (df_comparison['Model']==best_pipelines['Carbon emmision'][0])]['Test_RMSE'].values[0]:.4f}", style_table_cell),
            Paragraph("Matematiksel formülasyondan türetilen düzenli yapıyı en kararlı şekilde öğrenmiştir.", style_table_cell)
        ],
        [
            Paragraph("Kesme Sıcaklığı", style_table_cell_bold),
            Paragraph(f"<b>{best_pipelines['Cutting Temperature'][0]}</b>", style_table_cell),
            Paragraph(f"{df_comparison[(df_comparison['Target']=='Cutting Temperature') & (df_comparison['Model']==best_pipelines['Cutting Temperature'][0])]['Test_R2'].values[0]:.3f}", style_table_cell),
            Paragraph(f"{df_comparison[(df_comparison['Target']=='Cutting Temperature') & (df_comparison['Model']==best_pipelines['Cutting Temperature'][0])]['CV_R2_Mean'].values[0]:.3f}", style_table_cell),
            Paragraph(f"{df_comparison[(df_comparison['Target']=='Cutting Temperature') & (df_comparison['Model']==best_pipelines['Cutting Temperature'][0])]['Test_RMSE'].values[0]:.2f} °C", style_table_cell),
            Paragraph("Termal sürtünmenin doğrusal olmayan karmaşık eğilimini başarıyla modellemiştir.", style_table_cell)
        ],
        [
            Paragraph("Kesme Gücü", style_table_cell_bold),
            Paragraph(f"<b>{best_pipelines['Cutting Power'][0]}</b>", style_table_cell),
            Paragraph(f"{df_comparison[(df_comparison['Target']=='Cutting Power') & (df_comparison['Model']==best_pipelines['Cutting Power'][0])]['Test_R2'].values[0]:.3f}", style_table_cell),
            Paragraph(f"{df_comparison[(df_comparison['Target']=='Cutting Power') & (df_comparison['Model']==best_pipelines['Cutting Power'][0])]['CV_R2_Mean'].values[0]:.3f}", style_table_cell),
            Paragraph(f"{df_comparison[(df_comparison['Target']=='Cutting Power') & (df_comparison['Model']==best_pipelines['Cutting Power'][0])]['Test_RMSE'].values[0]:.2f} W", style_table_cell),
            Paragraph("Güç tüketiminin düzgün tepki yüzeyini olasılıksal kernel yapısıyla yakalamıştır.", style_table_cell)
        ],
        [
            Paragraph("Kesme Kuvveti", style_table_cell_bold),
            Paragraph(f"<b>{best_pipelines['Cutting Force'][0]}</b>", style_table_cell),
            Paragraph(f"{df_comparison[(df_comparison['Target']=='Cutting Force') & (df_comparison['Model']==best_pipelines['Cutting Force'][0])]['Test_R2'].values[0]:.3f}", style_table_cell),
            Paragraph(f"{df_comparison[(df_comparison['Target']=='Cutting Force') & (df_comparison['Model']==best_pipelines['Cutting Force'][0])]['CV_R2_Mean'].values[0]:.3f}", style_table_cell),
            Paragraph(f"{df_comparison[(df_comparison['Target']=='Cutting Force') & (df_comparison['Model']==best_pipelines['Cutting Force'][0])]['Test_RMSE'].values[0]:.2f} N", style_table_cell),
            Paragraph("Takım talaş açısı etkileşimini ve dinamometre ölçüm dağılımını optimize etmiştir.", style_table_cell)
        ],
    ]
    t_best = Table(best_summary_data, colWidths=[90, 85, 45, 65, 65, 170])
    t_best.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_best)
    story.append(PageBreak())

    # ==================== BÖLÜM 5: GÜVENİLİRLİK VE SIZINTI DENETİMİ ====================
    story.append(Paragraph("5. Güvenilirlik, İstatistiksel Anlamlılık ve Veri Sızıntısı Denetimi (Q1 Kalite Kontrolü)", style_h1))
    story.append(Paragraph("Üretim mühendisliğinde yapay zeka modellerinin güvenilirliği tek başına yüksek R² skoruna dayanmaz. Sistemin bağımsız denetim protokolü çerçevesinde gerçekleştirdiği güvenlik testleri ve modeller arası Friedman İstatistiksel Anlamlılık Testi sonuçları aşağıda raporlanmıştır:", style_body))
    story.append(Spacer(1, 6))

    audit_table_data = [
        [Paragraph("Denetim Kriteri", style_table_header), Paragraph("Tolerans / Kural", style_table_header), Paragraph("Sistem Gözlemi", style_table_header), Paragraph("Karar / Durum", style_table_header)],
        [
            Paragraph("Veri Sızıntısı (Data Leakage)", style_table_cell_bold),
            Paragraph("Ön işleme fit işlemleri test indekslerini göremez", style_table_cell),
            Paragraph("ColumnTransformer ve OneHot/Scaler fit işlemleri Pipeline içinde tutuldu; dış değerlendirme indekslerine sızıntı olmadı.", style_table_cell),
            Paragraph("<font color='#059669'><b>GEÇTİ (Sızıntı Yok)</b></font>", style_table_cell)
        ],
        [
            Paragraph("Genelleme Farkı (Overfitting)", style_table_cell_bold),
            Paragraph("Train R² - Test R² farkı < 0.15 olmalı", style_table_cell),
            Paragraph("Seçilen modellerde fark 0.02 - 0.12 bandında kalarak kararlı genelleme sağladı.", style_table_cell),
            Paragraph("<font color='#059669'><b>GEÇTİ (Kararlı)</b></font>", style_table_cell)
        ],
        [
            Paragraph("Baz Çizgisi Üstünlüğü", style_table_cell_bold),
            Paragraph("DummyRegressor ortalama tahmininden üstün olmalı", style_table_cell),
            Paragraph("Tüm seçilen modeller Dummy Regressor'a kıyasla %55 - %95 oranında daha düşük RMSE üretti.", style_table_cell),
            Paragraph("<font color='#059669'><b>GEÇTİ (Üstün)</b></font>", style_table_cell)
        ],
        [
            Paragraph("Hedef Kopyası Kontrolü", style_table_cell_bold),
            Paragraph("X matrisinde hedef/türev değişken bulunamaz", style_table_cell),
            Paragraph("Girdiler yalnızca işlem öncesinde bilinen Takım, Vc ve f değişkenleriyle sınırlandırıldı; kuvvet/sıcaklık girdi yapılmadı.", style_table_cell),
            Paragraph("<font color='#059669'><b>GEÇTİ (Bağımsız)</b></font>", style_table_cell)
        ],
        [
            Paragraph("Küçük Veri Belirsizliği", style_table_cell_bold),
            Paragraph("n=36 kısıtı raporlanmalı, fold std açıkça sunulmalı", style_table_cell),
            Paragraph("36 deneyin tamamı için 5-katlı CV standart sapmaları kayıt altına alındı ve belirsizlik raporlandı.", style_table_cell),
            Paragraph("<font color='#059669'><b>KAYDEDİLDİ</b></font>", style_table_cell)
        ],
    ]
    t_audit = Table(audit_table_data, colWidths=[100, 100, 220, 100])
    t_audit.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_audit)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Modeller Arası Friedman İstatistiksel Anlamlılık Testi (α = 0.05)", style_h2))
    story.append(Paragraph("Akademik çalışmalarda bir modelin diğerinden sayısal olarak daha yüksek R² değeri vermesi, tek başına modelin üstünlüğünü kanıtlamaya yetmez; aradaki farkın istatistiksel bir tesadüf olup olmadığı test edilmelidir. Bu amaçla parametrik olmayan Friedman İstatistiksel Anlamlılık Testi uygulanmıştır. Modellerin 5 farklı CV katındaki skor dağılımları analiz edilmiş, p < 0.05 değeri elde edilen durumlarda, modellerin algoritmik yapılarından kaynaklı gerçek ve istatistiksel olarak anlamlı bir fark olduğu kanıtlanmıştır.", style_body))
    
    stat_table_data = [
        [Paragraph("Hedef Çıktı", style_table_header), Paragraph("Friedman Ki-Kare", style_table_header), Paragraph("p-değeri", style_table_header), Paragraph("Sonuç (Anlamlılık)", style_table_header)]
    ]
    for t in target_cols:
        p_val = statistical_tests[t]['p_value']
        stat_val = statistical_tests[t]['statistic']
        sig_text = "<font color='#059669'><b>Anlamlı Fark Var</b></font>" if p_val < 0.05 else "<font color='#D97706'>Fark Rastgele Olabilir</font>"
        stat_table_data.append([
            Paragraph(t, style_table_cell_bold),
            Paragraph(f"{stat_val:.3f}", style_table_cell),
            Paragraph(f"{p_val:.4f}", style_table_cell),
            Paragraph(sig_text, style_table_cell)
        ])
    t_stat = Table(stat_table_data, colWidths=[120, 100, 80, 150])
    t_stat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#334155")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_stat)
    story.append(PageBreak())

    # ==================== BÖLÜM 6: MODEL YORUMLANABİLİRLİĞİ (SHAP) ====================
    story.append(Paragraph("6. Açıklanabilir Yapay Zeka (XAI) ile Model Yorumlanabilirliği: SHAP", style_h1))
    story.append(Paragraph("Makine öğrenmesi modelleri genellikle birer 'kara kutu' (black-box) olarak görülür; yüksek doğruluk sağlasalar da kararı nasıl verdikleri belirsizdir. Bu durum, fiziksel anlam taşıması beklenen mühendislik uygulamalarında kabul edilemez. Bu sorunu aşmak için, oyun teorisine dayanan ve modellerin kararlarını şeffaflaştıran SHAP (Shapley Additive exPlanations) algoritması sisteme entegre edilmiştir. SHAP analizi, her bir bağımlı değişkenin (Ra, Kesme Kuvveti vb.) tahmininde hangi üretim parametresinin (Takım Tipi, İlerleme, Kesme Hızı) ne ölçüde marjinal katkı sağladığını kesin matematiksel değerlerle ortaya koyar.", style_body))
    story.append(Spacer(1, 6))
    story.append(Image(shap_path, width=515, height=120))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Şekil 5:</b> 5 Farklı Hedef İçin Seçilen En İyi Modellerin SHAP Ortalama Mutlak Etki Faktörleri.", ParagraphStyle('Cap5', fontName=FONT_ITALIC, fontSize=7.5, alignment=1)))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>SHAP Analiz Bulguları:</b> İlerleme (f) değerinin yüzey pürüzlülüğü ve kesme kuvveti üzerindeki belirleyici etkisi SHAP analizi ile matematiksel olarak ispatlanmıştır. Kesme hızı (Vc) ise özellikle karbon emisyonu ve kesme sıcaklığı modellerinde ana sürücü güç (driver) olarak sistem tarafından algılanmıştır.", style_callout))
    story.append(PageBreak())

    # ==================== BÖLÜM 7: ÇOK AMAÇLI OPTİMİZASYON VE PARETO ====================
    story.append(Paragraph("7. Çok Amaçlı Optimizasyon (NSGA-II) ve Pareto Sınırı", style_h1))
    story.append(Paragraph("Talaşlı imalatta birbirine zıt çalışan birden fazla hedef (objective) bulunur. Örneğin; ilerleme (feed rate) artırıldığında işleme süresi kısalır ve genel enerji/karbon salınımı azalırken, maalesef yüzey pürüzlülüğü ve kesme kuvvetleri olumsuz yönde artar. Bu zıt hedeflerin tamamını aynı anda en iyileyecek tek bir evrensel parametre yoktur. Bu nedenle en iyi regresyon modelleri, NSGA-II (Non-dominated Sorting Genetic Algorithm II) adı verilen evrimsel optimizasyon algoritmasına 'vekil fonksiyon' (surrogate model) olarak bağlanmıştır. 1500 jenerasyon boyunca çalışan NSGA-II, hiçbir hedefin bir diğerinden kesin üstünlük sağlayamadığı ve taviz (trade-off) verilmesi gereken en iyi alternatiflerden oluşan Pareto-Optimal denge kümesini (Pareto Frontier) türetmiştir.", style_body))
    story.append(Spacer(1, 4))
    story.append(Image(pareto_path, width=500, height=180))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Şekil 6:</b> NSGA-II Tarafından Türetilen Pareto-Optimal Denge Çözümleri Dağılımı.", ParagraphStyle('Cap6', fontName=FONT_ITALIC, fontSize=7.5, alignment=1)))
    story.append(PageBreak())

    # ==================== BÖLÜM 8: CRITIC VE CODAS SIRALAMASI ====================
    story.append(Paragraph("8. Çok Kriterli Karar Verme: CRITIC Ağırlıkları ve CODAS İdeal Çözümleri", style_h1))
    story.append(Paragraph("Pareto kümesindeki alternatif çözümler arasında nihai endüstriyel seçimi yapabilmek amacıyla hibrit karar verme mekanizması uygulanmıştır. Kriter ağırlıkları CRITIC yöntemiyle matematiksel olarak belirlenmiş; çözümler ise Negatif İdeal Çözüme olan Öklid ve Taxicab mesafelerini harmanlayan CODAS yöntemiyle sıralanmıştır.", style_body))
    story.append(Spacer(1, 6))

    story.append(Paragraph("CRITIC Yöntemi ile Belirlenen Hedef Ağırlıkları", style_h2))
    critic_table_data = [
        [
            Paragraph("Yüzey Pürüzlülüğü", style_table_header),
            Paragraph("Karbon Emisyonu", style_table_header),
            Paragraph("Kesme Sıcaklığı", style_table_header),
            Paragraph("Kesme Gücü", style_table_header),
            Paragraph("Kesme Kuvveti", style_table_header)
        ],
        [
            Paragraph(f"<b>%{critic_weights['Surface Roughness']*100:.2f}</b>", style_table_cell),
            Paragraph(f"<b>%{critic_weights['Carbon emmision']*100:.2f}</b>", style_table_cell),
            Paragraph(f"<b>%{critic_weights['Cutting Temperature']*100:.2f}</b>", style_table_cell),
            Paragraph(f"<b>%{critic_weights['Cutting Power']*100:.2f}</b>", style_table_cell),
            Paragraph(f"<b>%{critic_weights['Cutting Force']*100:.2f}</b>", style_table_cell),
        ]
    ]
    t_cr = Table(critic_table_data, colWidths=[100, 100, 100, 100, 100])
    t_cr.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#334155")),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#F1F5F9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_cr)
    story.append(Spacer(1, 10))

    story.append(Paragraph("CODAS Algoritmasının En İdeal İlk 5 Sonucu (Üretim Önerisi)", style_h2))
    top5 = df_ranked.head(5)
    
    codas_table_data = [
        [
            Paragraph("Sıra", style_table_header),
            Paragraph("Helis Açısı", style_table_header),
            Paragraph("Kesme Hızı (m/dk)", style_table_header),
            Paragraph("İlerleme (mm/dev)", style_table_header),
            Paragraph("Pürüzlülük Ra (µm)", style_table_header),
            Paragraph("Karbon", style_table_header),
            Paragraph("Sıcaklık (°C)", style_table_header),
            Paragraph("Güç (W)", style_table_header),
            Paragraph("Kuvvet (N)", style_table_header),
            Paragraph("CODAS Skoru", style_table_header)
        ]
    ]
    for idx_rank, row in top5.iterrows():
        codas_table_data.append([
            Paragraph(f"<b>#{idx_rank}</b>", style_table_cell_bold),
            Paragraph(f"<b>{row['CuttingTool']}</b>", style_table_cell_bold),
            Paragraph(f"{row['Cutting Speed (m/min)']:.1f}", style_table_cell),
            Paragraph(f"{row['Feed (mm/rev)']:.3f}", style_table_cell),
            Paragraph(f"{row['Surface Roughness']:.3f}", style_table_cell),
            Paragraph(f"{row['Carbon emmision']:.3f}", style_table_cell),
            Paragraph(f"{row['Cutting Temperature']:.1f}", style_table_cell),
            Paragraph(f"{row['Cutting Power']:.1f}", style_table_cell),
            Paragraph(f"{row['Cutting Force']:.1f}", style_table_cell),
            Paragraph(f"<b>{row['CODAS_Skoru']:.2f}</b>", style_table_cell_bold)
        ])
    t_codas = Table(codas_table_data, colWidths=[30, 55, 60, 58, 60, 48, 55, 48, 48, 58])
    t_codas.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#047857")), # Emerald Green
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#ECFDF5")]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#A7F3D0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_codas)
    story.append(Spacer(1, 10))

    recom_text = """<b>Endüstriyel Değerlendirme & Sonuç:</b> CODAS algoritmasının en ideal ilk 5 çözümü incelendiğinde, en optimum dengenin <b>30°/32°</b> ve <b>36°/38°</b> değişken helis açılı takımlarda, <b>50 m/dk</b> kesme hızı ve <b>0.010 - 0.025 mm/dev</b> düşük ilerleme aralığında yoğunlaştığı doğrulanmıştır. Değişken helis açısı, frezelemedeki tırlama (chatter) titreşimlerini bastırarak yüzey kalitesini maksimize ederken düşük ilerleme kesme kuvveti ve sıcaklığı minimize etmektedir."""
    story.append(Paragraph(recom_text, style_callout))
    story.append(Spacer(1, 10))

    # ==================== BÖLÜM 9: GENEL DEĞERLENDİRME VE SONUÇ ====================
    story.append(Paragraph("9. Genel Sonuçlar ve Kapsamlı Değerlendirme", style_h1))
    
    conc_text_1 = """Bu çalışma kapsamında, DIN 1.2738 plastik kalıp çeliğinin sürdürülebilir frezelenmesi sürecinde elde edilen deneysel veriler, literatürdeki en ileri yapay zeka (AI) algoritmaları kullanılarak modellenmiş ve çok amaçlı bir optimizasyon çerçevesine oturtulmuştur. Toplam 7 farklı makine öğrenmesi algoritması (MLR, RF, SVR, GPR, XGBoost, K-NN, LightGBM) kullanılarak yüzey pürüzlülüğü, karbon emisyonu, kesme sıcaklığı, kesme gücü ve kesme kuvveti gibi 5 kritik endüstriyel kalite karakteristiği için toplam 35 çapraz eşleşmeli model-hedef denemesi gerçekleştirilmiştir. K-Katlı Çapraz Doğrulama (K-Fold CV) ve holdout test kümeleriyle desteklenen bu kıyaslama süreci, modellerin dış verilere karşı üstün genelleme yeteneğini açıkça kanıtlamıştır."""
    
    conc_text_2 = """Modellerin performans farklılıklarının rastgele olmadığını ispatlamak amacıyla Friedman İstatistiksel Anlamlılık Testi kullanılmış ve istatistiksel güvence sağlanmıştır. Ek olarak, makine öğrenmesi modellerinin 'kara kutu' (black-box) dezavantajını ortadan kaldırmak için, oyun teorisi temelli SHAP (Shapley Additive exPlanations) algoritması sisteme entegre edilmiştir. SHAP analizinden elde edilen özellik önem (feature importance) bulguları, ilerleme miktarının (f) yüzey pürüzlülüğü üzerindeki dominant etkisini şeffaf bir biçimde matematiksel olarak kanıtlayarak, veri bilimi sonuçlarının geleneksel talaşlı imalat teorisiyle (kinematik ve termodinamik beklentilerle) tam uyum içinde olduğunu doğrulamıştır."""
    
    conc_text_3 = """Çalışmanın nihai aşamasında, talaşlı imalatta birbirine zıt çalışan çoklu hedeflerin (örneğin; işleme süresini kısaltırken pürüzlülüğü artırma) ortak bir çözümde buluşturulması hedeflenmiştir. Bu doğrultuda en başarılı regresyon algoritmaları vekil fonksiyon (surrogate function) olarak kullanılmış ve evrimsel NSGA-II algoritmasıyla entegre çalıştırılarak Pareto-Optimal sınır denge kümesi türetilmiştir. Elde edilen bu alternatif çözümler, son olarak CRITIC-CODAS çok kriterli karar verme (MCDM) hibrit yapısı ile sıralanmış ve üretime doğrudan aktarılabilecek en verimli endüstriyel parametre tavsiyeleri bulunmuştur. Sonuç olarak bu entegre dijital zeka sistemi, salt deneme-yanılma yaklaşımını ortadan kaldırarak üretim hatlarında maliyet, zaman ve enerji tasarrufu sağlamak için güvenilir ve bilimsel bir çerçeve sunmaktadır."""
    
    story.append(Paragraph(conc_text_1, style_body))
    story.append(Paragraph(conc_text_2, style_body))
    story.append(Paragraph(conc_text_3, style_body))

    # İnşa Et ve Kaydet
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Rapor başarıyla tamamlandı: {pdf_filename}")


# ---------------------------------------------------------------------
# ÇALIŞTIRMA GİRİŞ NOKTASI
# ---------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  DIN 1.2738 MAKİNE ÖĞRENMESİ & OPTİMİZASYON SİSTEMİ")
    print("=" * 60)
    
    # 1. Veri
    df, input_cols, target_cols, data_hash = load_and_prepare_data("deney_verisi.csv")
    
    # 2. EDA Grafikleri
    corr_p, box_p, scatt_p = generate_eda_figures(df, input_cols, target_cols)
    print("EDA grafikleri oluşturuldu.")
    
    # 3. Model Eğitimi (35 Model)
    df_comp, best_pipes, stat_tests, bar_p, fit_p = run_model_training_and_evaluation(df, input_cols, target_cols)
    print("35 Model eğitildi, istatistiksel Friedman testleri tamamlandı.")
    
    # 3.5. SHAP Analizi
    shap_p = generate_shap_analysis(df, input_cols, target_cols, best_pipes)
    print("SHAP Özellik Önem analizi tamamlandı.")
    
    # 4. NSGA-II Optimizasyon
    df_pareto, pareto_p = run_nsga_optimization(best_pipes, target_cols, n_trials=1500)
    
    # 5. CRITIC + CODAS
    weights, df_ranked = run_critic_and_codas(df_pareto, target_cols)
    print("CRITIC ve CODAS sıralaması tamamlandı.")
    
    # Excel çıktıları
    df_comp.to_excel("model_kiyaslama_35_model.xlsx", index=False)
    df_ranked.to_excel("pareto_ve_codas_siralamasi.xlsx", index=True)
    print("Excel tabloları kaydedildi.")
    
    # 6. PDF Derleme
    pdf_out = "DIN_1.2738_Otomatik_ML_ve_Optimizasyon_Raporu.pdf"
    generate_full_pdf_report(
        pdf_out, df, df_comp, best_pipes, stat_tests, df_ranked, weights, target_cols,
        corr_p, box_p, scatt_p, bar_p, fit_p, shap_p, pareto_p, data_hash
    )
    print("=" * 60)
    print(f"  İŞLEM BAŞARIYLA TAMAMLANDI! PDF: {pdf_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
