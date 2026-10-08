import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import datetime
import warnings
import optuna

warnings.filterwarnings('ignore')

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
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

st.set_page_config(page_title="Otomatik ML Sistemi", layout="wide")

# =====================================================================
# YARDIMCI FONKSİYONLAR
# =====================================================================

def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

def get_models():
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
    models = get_models()
    results = {}
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred_train = model.predict(X_train_sc)
        y_pred_test = model.predict(X_test_sc)
        cv_scores = cross_val_score(model, X_train_sc, y_train, cv=3, scoring='neg_root_mean_squared_error')

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
            "cv_rmse_mean": -cv_scores.mean(),
            "cv_rmse_std": cv_scores.std(),
        }
    return results

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf

# =====================================================================
# ÇOK AMAÇLI OPTİMİZASYON (NSGA-II & CRITIC-CODAS)
# =====================================================================

def calculate_critic_weights(df_targets):
    """CRITIC algoritması ile ağırlıkları hesaplar."""
    if len(df_targets) < 3:
        return pd.Series([1.0/len(df_targets.columns)]*len(df_targets.columns), index=df_targets.columns)
    
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
    conflict = 1 - corr
    C = std * conflict.sum(axis=1)
    
    if C.sum() == 0:
        return pd.Series([1.0/len(df_targets.columns)]*len(df_targets.columns), index=df_targets.columns)
        
    return C / C.sum()

def apply_codas(df_targets, weights, tau=0.02):
    """CODAS algoritması ile Pareto çözümlerini sıralar."""
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

# =====================================================================
# ANA UYGULAMA
# =====================================================================

def main():
    st.title("🔬 Otomatik ML & Çok Amaçlı Optimizasyon (NSGA & CODAS)")
    st.markdown("**1. Veriyi yükleyin → 2. Sütunları seçin → 3. Modelleri eğitip Pareto Optimizasyonu yapın.**")
    st.divider()

    uploaded_file = st.file_uploader("📁 Veri Setini Yükle (CSV veya Excel)", type=["csv", "xlsx"])

    if uploaded_file is None:
        st.info("Lütfen veri setinizi yükleyin.")
        return

    df = load_data(uploaded_file)
    st.subheader("📊 Yüklenen Veri")
    st.dataframe(df, use_container_width=True)

    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.warning(f"{missing.sum()} eksik değer bulundu. Temizleniyor...")
        df = df.dropna()

    st.divider()
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    inputs = st.multiselect("🟢 Girdi Değişkenleri (Optimizasyon Karar Değişkenleri)", all_numeric)
    remaining = [c for c in all_numeric if c not in inputs]
    targets = st.multiselect("🔴 Hedef Değişkenler (Minimize Edilecek Amaçlar)", remaining)

    if not inputs or not targets:
        st.warning("Lütfen girdi ve hedef sütunları seçin.")
        return

    if not st.button("🚀 TÜM SİSTEMİ ÇALIŞTIR (Eğitim + NSGA Optimizasyonu)", type="primary", use_container_width=True):
        return

    st.divider()
    # ================= BÖLÜM 1: KEŞİF =================
    st.header("1. Veri Keşfi")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Korelasyon")
        fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[inputs + targets].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax_corr)
        st.pyplot(fig_corr)
    with col2:
        st.subheader("Dağılımlar")
        fig_dist, axes_dist = plt.subplots(1, len(targets), figsize=(4*len(targets), 3))
        if len(targets) == 1: axes_dist = [axes_dist]
        for i, col in enumerate(targets):
            sns.histplot(df[col], kde=True, ax=axes_dist[i], color='coral')
            axes_dist[i].set_title(col)
        fig_dist.tight_layout()
        st.pyplot(fig_dist)

    # ================= BÖLÜM 2: EĞİTİM =================
    st.divider()
    st.header("2. Model Eğitimi")
    
    all_results = {}
    best_models_for_opt = {}
    master_scaler = StandardScaler()
    master_scaler.fit(df[inputs].values)

    progress = st.progress(0)
    for t_idx, target_col in enumerate(targets):
        X = df[inputs].values
        y = df[target_col].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Eğit
        results = train_and_evaluate(X_train, X_test, y_train, y_test, master_scaler)
        all_results[target_col] = results
        
        # En iyi modeli seç
        best_name = max(results.items(), key=lambda x: x[1]["test_r2"])[0]
        best_models_for_opt[target_col] = results[best_name]["model"]
        
        st.success(f"**{target_col}** için en iyi model: {best_name} (Test R²: {results[best_name]['test_r2']:.3f})")
        progress.progress((t_idx + 1) / len(targets))

    # ================= BÖLÜM 3: ÇOK AMAÇLI OPTİMİZASYON =================
    st.divider()
    st.header("3. Çok Amaçlı Optimizasyon (NSGA & CODAS)")
    st.info("Eğitilen en iyi modeller surrogate fonksiyon olarak kullanılarak Pareto optimal çözümler aranıyor...")
    
    # Optuna ayarları
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bounds = {}
    for inp in inputs:
        n_unique = df[inp].nunique()
        if n_unique < 5:
            bounds[inp] = {"type": "cat", "vals": df[inp].unique().tolist()}
        else:
            bounds[inp] = {"type": "float", "min": float(df[inp].min()), "max": float(df[inp].max())}

    def objective(trial):
        x_dict = {}
        for inp in inputs:
            if bounds[inp]["type"] == "cat":
                x_dict[inp] = trial.suggest_categorical(inp, bounds[inp]["vals"])
            else:
                x_dict[inp] = trial.suggest_float(inp, bounds[inp]["min"], bounds[inp]["max"])
                
        x_arr = pd.DataFrame([x_dict])[inputs].values
        x_scaled = master_scaler.transform(x_arr)
        
        preds = []
        for t in targets:
            preds.append(float(best_models_for_opt[t].predict(x_scaled)[0]))
        return tuple(preds)

    study = optuna.create_study(directions=["minimize"] * len(targets))
    study.optimize(objective, n_trials=1000)

    # Pareto setini çıkar
    pareto_data = []
    for pt in study.best_trials:
        row = pt.params.copy()
        for i, t in enumerate(targets):
            row[t] = pt.values[i]
        pareto_data.append(row)
        
    df_pareto = pd.DataFrame(pareto_data).drop_duplicates()
    
    if len(df_pareto) == 0:
        st.error("Pareto sınırı bulunamadı.")
    else:
        st.write(f"NSGA algoritması {len(df_pareto)} adet Pareto-optimal denge çözümü buldu.")
        
        # CRITIC
        df_targets_only = df_pareto[targets]
        weights = calculate_critic_weights(df_targets_only)
        
        st.subheader("CRITIC Yöntemi ile Belirlenen Hedef Ağırlıkları")
        st.dataframe(pd.DataFrame(weights, columns=["Ağırlık (%)"]).apply(lambda x: x*100).round(2).T)
        
        # CODAS
        scores = apply_codas(df_targets_only, weights)
        df_pareto["CODAS_Skoru"] = scores
        df_pareto = df_pareto.sort_values(by="CODAS_Skoru", ascending=False).reset_index(drop=True)
        df_pareto.index = df_pareto.index + 1
        
        st.subheader("CODAS Algoritmasının En İdeal İlk 5 Sonucu")
        st.dataframe(df_pareto.head(5).style.background_gradient(subset=targets, cmap="viridis_r"))

if __name__ == "__main__":
    main()
