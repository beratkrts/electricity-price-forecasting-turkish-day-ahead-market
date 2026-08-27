import nbformat as nbf

nb = nbf.v4.new_notebook()

md_intro = """# PTF Sıfır Fiyat (Zero-Price) Hibrit Sınıflandırma Modeli Ar-Ge
Bahar aylarındaki fiyat çökmelerini öngörmek için oluşturulan Classifier + Regressor hibrit model test ortamı.
"""

code_setup = """import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path.cwd().parent))
from scripts.predict_daily_pipeline import load_all_historical_data
from src.features.feature_engineering import build_robust_features, get_feature_columns

print("Veriler DB'den yükleniyor...")
df_raw = load_all_historical_data()
df_feat = build_robust_features(df_raw)
"""

code_prepare = """# Hedef Değişkenleri Belirleme
target_col = 'mcp_price_usd'

# Eşik Değer (Örn: 5 Doların altını "Sıfır Fiyat Çöküşü" olarak tanımlıyoruz)
ZERO_PRICE_THRESHOLD = 5.0

features = get_feature_columns('robust', df_feat)
df = df_feat.dropna(subset=features + [target_col]).copy()

# Sınıflandırma için yeni Hedef: Fiyat 5 Doların altında mı? (1=Evet, 0=Hayır)
df['is_zero_price'] = (df[target_col] <= ZERO_PRICE_THRESHOLD).astype(int)

# Train/Test Split (Son 6 Ay Test)
train_size = len(df) - (180 * 24)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print(f"Toplam Çöküş Saati Oranı: %{df['is_zero_price'].mean()*100:.2f}")
"""

code_classifier = """# 1. CLASSIFIER MODELİ (Sıfır Olacak mı?)
clf = lgb.LGBMClassifier(
    n_estimators=100, 
    learning_rate=0.05, 
    max_depth=5, 
    class_weight='balanced', # Sınıflar dengesiz olduğu için ağırlıklandırıyoruz
    random_state=42
)

clf.fit(train_df[features], train_df['is_zero_price'])

# Test Seti Tahmini
test_df['pred_is_zero_prob'] = clf.predict_proba(test_df[features])[:, 1]
test_df['pred_is_zero_class'] = (test_df['pred_is_zero_prob'] > 0.5).astype(int)

print("--- CLASSIFIER BAŞARISI (Sıfır Fiyatı Yakalama) ---")
print(classification_report(test_df['is_zero_price'], test_df['pred_is_zero_class']))
"""

code_regressor = """# 2. REGRESSOR MODELİ (Normal Fiyat Ne Olacak?)
# Regressor'ü eğitirken sadece NORMAL (Sıfır olmayan) günleri kullanarak beynini temizliyoruz!
train_normal = train_df[train_df['is_zero_price'] == 0]

reg = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
reg.fit(train_normal[features], train_normal[target_col])

test_df['pred_normal_price'] = reg.predict(test_df[features])
"""

code_hybrid = """# 3. HİBRİT BİRLEŞTİRME VE TEST
# KURAL: Eğer Classifier "Sıfır" (1) dediyse, fiyatı 0.0 yap. Değilse Regressor'ün tahminini kullan.
test_df['pred_hybrid_price'] = np.where(
    test_df['pred_is_zero_class'] == 1, 
    0.0, 
    test_df['pred_normal_price']
)

# Kıyaslama için eski "Sadece Regressor" yaklaşımı (tüm veride eğitilmiş)
reg_old = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
reg_old.fit(train_df[features], train_df[target_col])
test_df['pred_old_price'] = reg_old.predict(test_df[features])

def wape(y_t, y_p):
    return (np.abs(y_t - y_p).sum() / y_t.sum()) * 100

print(f"Eski Standart Model WAPE: {wape(test_df[target_col], test_df['pred_old_price']):.2f}%")
print(f"YENİ HİBRİT MODEL WAPE: {wape(test_df[target_col], test_df['pred_hybrid_price']):.2f}%")
"""

nb.cells = [
    nbf.v4.new_markdown_cell(md_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_code_cell(code_prepare),
    nbf.v4.new_code_cell(code_classifier),
    nbf.v4.new_code_cell(code_regressor),
    nbf.v4.new_code_cell(code_hybrid)
]

with open('eda/hybrid_ptf_classifier.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook 'eda/hybrid_ptf_classifier.ipynb' başarıyla oluşturuldu!")
