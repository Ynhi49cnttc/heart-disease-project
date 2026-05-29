"""
Heart Disease Risk Prediction — Streamlit Demo
Đề tài: Phân tích các chỉ số sức khỏe và dự đoán nguy cơ mắc bệnh tim

Cài đặt:
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn scipy

Chạy:
    streamlit run app_heart_disease.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                              classification_report, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        color: #C0392B; text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem; color: #555; text-align: center; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 4px solid #E8634C;
    }
    .risk-high   { background:#fdecea; color:#c0392b;
                   border-radius:10px; padding:1rem; text-align:center; font-size:1.3rem; font-weight:700; }
    .risk-low    { background:#eafaf1; color:#1e8449;
                   border-radius:10px; padding:1rem; text-align:center; font-size:1.3rem; font-weight:700; }
    .section-header { font-size:1.3rem; font-weight:700; color:#2C3E50;
                      border-bottom:2px solid #E8634C; padding-bottom:4px; margin-top:1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING & MODEL TRAINING (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Đang tải dataset UCI Heart Disease...")
def load_data():
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "heart-disease/processed.cleveland.data")
    columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
                'thalach','exang','oldpeak','slope','ca','thal','target']
    df = pd.read_csv(url, header=None, names=columns, na_values='?')
    df['target'] = (df['target'] > 0).astype(int)
    df['ca']   = df['ca'].fillna(df['ca'].mode()[0]).astype(int)
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0]).astype(int)
    for col in ['sex','cp','fbs','restecg','exang','slope','ca','thal']:
        df[col] = df[col].astype(int)
    return df

@st.cache_resource(show_spinner="Đang huấn luyện mô hình...")
def train_models(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting'  : GradientBoostingClassifier(random_state=42),
        'SVM'                : SVC(probability=True, random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    trained = {}
    for name, m in models.items():
        m.fit(X_train_sc, y_train)
        cv_auc = cross_val_score(m, X_train_sc, y_train, cv=cv, scoring='roc_auc').mean()
        cv_acc = cross_val_score(m, X_train_sc, y_train, cv=cv, scoring='accuracy').mean()
        y_pred      = m.predict(X_test_sc)
        y_pred_prob = m.predict_proba(X_test_sc)[:, 1]
        trained[name] = {
            'model': m,
            'cv_auc': cv_auc,
            'cv_acc': cv_acc,
            'test_auc': roc_auc_score(y_test, y_pred_prob),
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob,
        }
    best_name = max(trained, key=lambda k: trained[k]['test_auc'])
    return trained, scaler, X_test, y_test, best_name, X.columns.tolist()


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
df = load_data()
trained, scaler, X_test, y_test, best_name, feature_cols = train_models(df)

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/R_logo.svg/1200px-R_logo.svg.png",
             width=1)  # placeholder spacer
    st.markdown("## 🫀 Heart Disease App")
    st.markdown("**Đề tài:** Phân tích chỉ số sức khỏe & Dự đoán nguy cơ bệnh tim")
    st.markdown("**Dataset:** UCI Cleveland (303 mẫu, 14 features)")
    st.divider()
    page = st.radio("Chọn trang", [
        "🏠 Tổng quan",
        "📊 Phân tích EDA",
        "🔬 Kết quả kiểm định",
        "🤖 Mô hình ML",
        "🩺 Dự đoán cá nhân",
    ])

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
BLUE  = '#4C9BE8'
RED   = '#E8634C'
COLORS = [BLUE, RED]

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: TỔNG QUAN
# ─────────────────────────────────────────────
if page == "🏠 Tổng quan":
    st.markdown('<div class="main-title">🫀 Heart Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Phân tích chỉ số sức khỏe & Dự đoán nguy cơ mắc bệnh tim</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng mẫu",       f"{len(df):,}")
    c2.metric("Features",        "13")
    c3.metric("Có bệnh tim",    f"{df['target'].sum()} ({df['target'].mean():.1%})")
    c4.metric("Không bệnh tim", f"{(df['target']==0).sum()} ({(df['target']==0).mean():.1%})")

    st.divider()
    section("📋 Mô tả Dataset")
    feature_info = pd.DataFrame({
        'Feature':    ['age','sex','cp','trestbps','chol','fbs','restecg',
                       'thalach','exang','oldpeak','slope','ca','thal'],
        'Tên đầy đủ':['Tuổi','Giới tính','Loại đau ngực','Huyết áp nghỉ',
                      'Cholesterol','Đường huyết lúc đói','Kết quả ECG',
                      'Nhịp tim tối đa','Đau ngực khi gắng sức','ST depression',
                      'Độ dốc ST','Số mạch chính','Thalassemia'],
        'Loại':       ['Numeric','Binary','Categorical','Numeric','Numeric','Binary',
                       'Categorical','Numeric','Binary','Numeric',
                       'Categorical','Categorical','Categorical'],
    })
    st.dataframe(feature_info, use_container_width=True, hide_index=True)

    section("📄 Dữ liệu mẫu")
    st.dataframe(df.head(10), use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: EDA
# ─────────────────────────────────────────────
elif page == "📊 Phân tích EDA":
    st.markdown("## 📊 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Phân phối biến Numeric", "Biến Categorical", "Correlation"])

    numeric_cols = ['age','trestbps','chol','thalach','oldpeak']

    with tab1:
        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        for i, col in enumerate(numeric_cols):
            axes[0, i].hist(df[col], bins=20, color=BLUE, edgecolor='white', density=True, alpha=0.8)
            df[col].plot.kde(ax=axes[0, i], color='#185FA5', linewidth=2)
            axes[0, i].set_title(col, fontweight='bold')
            for tv, color, label in zip([0,1], COLORS, ['Không bệnh','Có bệnh']):
                df[df['target']==tv][col].plot.kde(ax=axes[1,i], color=color, linewidth=2, label=label)
            axes[1, i].set_title(f'{col} theo nhóm', fontweight='bold')
            if i == 4:
                axes[1, i].legend(fontsize=8)
        plt.suptitle('Phân phối các biến Numeric', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        section("Thống kê mô tả")
        st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)

    with tab2:
        cat_cols = ['sex','cp','exang','slope','ca','thal']
        cat_labels = {
            'sex':   {0:'Nữ', 1:'Nam'},
            'cp':    {0:'Không đau', 1:'Điển hình', 2:'Không điển hình', 3:'Không TC'},
            'exang': {0:'Không', 1:'Có'},
            'slope': {0:'Dốc lên', 1:'Phẳng', 2:'Dốc xuống'},
            'ca':    {0:'0 mạch', 1:'1 mạch', 2:'2 mạch', 3:'3 mạch'},
            'thal':  {1:'Bình thường', 2:'Fixed defect', 3:'Reversible'},
        }
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        axes = axes.flatten()
        for i, col in enumerate(cat_cols):
            data = df.groupby([col,'target']).size().unstack(fill_value=0)
            if col in cat_labels:
                data.index = [cat_labels[col].get(idx, str(idx)) for idx in data.index]
            data.columns = ['Không bệnh','Có bệnh']
            data.plot(kind='bar', ax=axes[i], color=COLORS, edgecolor='white', width=0.7)
            axes[i].set_title(col.upper(), fontweight='bold')
            axes[i].tick_params(axis='x', rotation=30)
            axes[i].legend(fontsize=8)
        plt.suptitle('Phân phối Categorical theo Target', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        fig, ax = plt.subplots(figsize=(11, 9))
        mask = np.triu(np.ones_like(df.corr(), dtype=bool))
        sns.heatmap(df.corr(), mask=mask, annot=True, fmt='.2f',
                    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5, ax=ax)
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        section("Tương quan với Target")
        target_corr = (df.corr()['target'].drop('target')
                       .sort_values(key=abs, ascending=False))
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        colors_bar = [RED if v > 0 else BLUE for v in target_corr]
        ax2.barh(target_corr.index, target_corr.values, color=colors_bar, height=0.6)
        ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax2.set_title('Tương quan từng feature với Target', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()


# ─────────────────────────────────────────────
# PAGE: KIỂM ĐỊNH THỐNG KÊ
# ─────────────────────────────────────────────
elif page == "🔬 Kết quả kiểm định":
    st.markdown("## 🔬 Kết quả Kiểm định Thống kê")

    from scipy.stats import ttest_ind, levene, chi2_contingency, shapiro

    def run_ttest(col, direction='two-sided'):
        g0 = df[df['target']==0][col].dropna()
        g1 = df[df['target']==1][col].dropna()
        _, lev_p = levene(g0, g1)
        t_stat, p_two = ttest_ind(g0, g1, equal_var=(lev_p > 0.05))
        p_one = p_two / 2
        d = abs(g0.mean() - g1.mean()) / np.sqrt((g0.std()**2 + g1.std()**2)/2)
        return g0.mean(), g1.mean(), t_stat, p_one, d

    def run_chi2(col):
        ct = pd.crosstab(df[col], df['target'])
        chi2, p, dof, _ = chi2_contingency(ct)
        v = np.sqrt(chi2 / (ct.sum().sum() * (min(ct.shape)-1)))
        return chi2, p, dof, v

    tab1, tab2, tab3 = st.tabs(["T-test (H1–H3)", "Chi-square (H4–H5)", "Bootstrap CI"])

    with tab1:
        for col, h, desc in [
            ('thalach','H1','Nhịp tim tối đa (bpm)'),
            ('oldpeak', 'H2','ST Depression'),
            ('age',     'H3','Tuổi'),
        ]:
            m0, m1, t, p, d = run_ttest(col)
            sig = "✅ Bác bỏ H₀" if p < 0.05 else "❌ Không bác bỏ H₀"
            effect = "Rất lớn" if d >= 0.8 else ("Lớn" if d >= 0.5 else "Trung bình")
            with st.expander(f"**{h}: {desc}** — {sig}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean (Không bệnh)", f"{m0:.2f}")
                c2.metric("Mean (Có bệnh)",    f"{m1:.2f}",
                          delta=f"{m1-m0:+.2f}", delta_color="inverse")
                c3.metric("p-value (one-sided)", f"{p:.2e}")
                c4.metric(f"Cohen's d ({effect})", f"{d:.3f}")

                fig, ax = plt.subplots(figsize=(8, 3))
                for tv, color, label in zip([0,1], COLORS, ['Không bệnh','Có bệnh']):
                    df[df['target']==tv][col].plot.kde(ax=ax, color=color, linewidth=2, label=label)
                ax.axvline(m0, color=BLUE, linestyle='--', alpha=0.7)
                ax.axvline(m1, color=RED,  linestyle='--', alpha=0.7)
                ax.set_title(f'KDE — {col}', fontweight='bold')
                ax.legend(); ax.grid(alpha=0.3)
                st.pyplot(fig); plt.close()

    with tab2:
        for col, h, desc in [('sex','H4','Giới tính'), ('cp','H5','Loại đau ngực')]:
            chi2, p, dof, v = run_chi2(col)
            sig = "✅ Bác bỏ H₀" if p < 0.05 else "❌ Không bác bỏ H₀"
            effect = "Mạnh" if v >= 0.5 else ("Trung bình" if v >= 0.3 else "Yếu")
            with st.expander(f"**{h}: {desc}** — {sig}", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Chi² statistic", f"{chi2:.3f}")
                c2.metric("p-value",        f"{p:.2e}")
                c3.metric(f"Cramér's V ({effect})", f"{v:.4f}")
                ct = pd.crosstab(df[col], df['target'])
                ct.columns = ['Không bệnh','Có bệnh']
                st.dataframe(ct, use_container_width=True)

    with tab3:
        np.random.seed(42)
        R = 2000
        boot_data = []
        for col in ['thalach','oldpeak','age']:
            for tv, label in [(0,'Không bệnh'),(1,'Có bệnh')]:
                data = df[df['target']==tv][col].dropna().values
                boot_means = [np.random.choice(data, len(data), replace=True).mean()
                              for _ in range(R)]
                boot_data.append({
                    'feature': col, 'group': label,
                    'mean': data.mean(),
                    'ci_lower': np.percentile(boot_means, 2.5),
                    'ci_upper': np.percentile(boot_means, 97.5),
                })

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        color_map = {'Không bệnh': BLUE, 'Có bệnh': RED}
        for ax, col in zip(axes, ['thalach','oldpeak','age']):
            rows = [r for r in boot_data if r['feature'] == col]
            for j, row in enumerate(rows):
                c = color_map[row['group']]
                ax.scatter(row['mean'], j, color=c, s=80, zorder=3)
                ax.plot([row['ci_lower'], row['ci_upper']], [j,j],
                        color=c, linewidth=3, alpha=0.8)
                ax.text(row['ci_upper']+0.3, j,
                        f"{row['mean']:.1f} [{row['ci_lower']:.1f}, {row['ci_upper']:.1f}]",
                        va='center', fontsize=9)
            ax.set_yticks([0,1])
            ax.set_yticklabels(['Không bệnh','Có bệnh'])
            ax.set_title(col.upper(), fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        plt.suptitle('Bootstrap 95% CI — So sánh 2 nhóm', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────
# PAGE: MÔ HÌNH ML
# ─────────────────────────────────────────────
elif page == "🤖 Mô hình ML":
    st.markdown("## 🤖 So sánh & Đánh giá Mô hình ML")

    # So sánh models
    section("So sánh Cross-Validation (5-Fold)")
    rows = []
    for name, info in trained.items():
        rows.append({
            'Model': name,
            'CV Accuracy': f"{info['cv_acc']:.4f}",
            'CV AUC':      f"{info['cv_auc']:.4f}",
            'Test AUC':    f"{info['test_auc']:.4f}",
            'Chọn': '✅ Best' if name == best_name else '',
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.success(f"**Model tốt nhất: {best_name}** (Test AUC = {trained[best_name]['test_auc']:.4f})")

    # ROC Curves cho tất cả model
    section("ROC Curve — Tất cả mô hình")
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, info in trained.items():
        fpr, tpr, _ = roc_curve(y_test, info['y_pred_prob'])
        ax.plot(fpr, tpr, linewidth=2,
                label=f"{name} (AUC={info['test_auc']:.3f})",
                linestyle='--' if name != best_name else '-')
    ax.plot([0,1],[0,1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Confusion Matrix của best model
    section(f"Confusion Matrix — {best_name}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cm = confusion_matrix(y_test, trained[best_name]['y_pred'])
    ConfusionMatrixDisplay(cm, display_labels=['Không bệnh','Có bệnh']).plot(
        ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title('Confusion Matrix', fontweight='bold')

    # Classification report as bar
    report = classification_report(y_test, trained[best_name]['y_pred'],
                                   target_names=['Không bệnh','Có bệnh'],
                                   output_dict=True)
    metrics = ['precision','recall','f1-score']
    x = np.arange(len(metrics))
    w = 0.3
    for i, cls in enumerate(['Không bệnh','Có bệnh']):
        vals = [report[cls][m] for m in metrics]
        axes[1].bar(x + i*w, vals, width=w, label=cls,
                    color=COLORS[i], edgecolor='white')
    axes[1].set_xticks(x + w/2); axes[1].set_xticklabels(metrics)
    axes[1].set_ylim(0, 1.1); axes[1].legend()
    axes[1].set_title('Precision / Recall / F1', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Feature Importance
    section("Feature Importance (Random Forest)")
    rf = trained['Random Forest']['model']
    imp_df = pd.DataFrame({
        'Feature':    feature_cols,
        'Importance': rf.feature_importances_,
    }).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    median_imp = imp_df['Importance'].median()
    colors_imp = [RED if v > median_imp else BLUE for v in imp_df['Importance']]
    ax.barh(imp_df['Feature'], imp_df['Importance'], color=colors_imp, height=0.6)
    ax.set_title('Feature Importance — Random Forest', fontweight='bold')
    ax.set_xlabel('Gini Importance')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────
# PAGE: DỰ ĐOÁN CÁ NHÂN
# ─────────────────────────────────────────────
elif page == "🩺 Dự đoán cá nhân":
    st.markdown("## 🩺 Dự đoán nguy cơ bệnh tim")
    st.info("Nhập thủ công từng bệnh nhân **hoặc** upload file CSV để dự đoán hàng loạt. Kết quả chỉ mang tính tham khảo — không thay thế chẩn đoán y khoa.")

    input_tab1, input_tab2 = st.tabs(["✍️ Nhập thủ công", "📂 Upload CSV (hàng loạt)"])

    # ── TAB 2: UPLOAD CSV ──────────────────────────────────────────
    with input_tab2:
        section("📂 Upload file CSV để dự đoán hàng loạt")

        # Hướng dẫn + nút tải template
        with st.expander("ℹ️ Hướng dẫn định dạng file CSV", expanded=True):
            st.markdown("""
File CSV cần có **13 cột** theo đúng thứ tự sau (không cần cột `target`):

| Cột | Mô tả | Giá trị hợp lệ |
|-----|--------|----------------|
| `age` | Tuổi | 20–100 |
| `sex` | Giới tính | 0 = Nữ, 1 = Nam |
| `cp` | Loại đau ngực | 0, 1, 2, 3 |
| `trestbps` | Huyết áp nghỉ (mmHg) | 80–220 |
| `chol` | Cholesterol (mg/dl) | 100–600 |
| `fbs` | Đường huyết > 120? | 0 = Không, 1 = Có |
| `restecg` | Kết quả ECG | 0, 1, 2 |
| `thalach` | Nhịp tim tối đa (bpm) | 60–220 |
| `exang` | Đau ngực khi gắng sức | 0 = Không, 1 = Có |
| `oldpeak` | ST Depression | 0.0–7.0 |
| `slope` | Độ dốc ST | 0, 1, 2 |
| `ca` | Số mạch chính | 0, 1, 2, 3 |
| `thal` | Thalassemia | 1, 2, 3 |
""")
            # Tạo template CSV cho người dùng tải về
            template_df = pd.DataFrame([
                [63,1,3,145,233,1,0,150,0,2.3,0,0,1],
                [37,1,2,130,250,0,1,187,0,3.5,0,0,2],
                [41,0,1,130,204,0,0,172,0,1.4,2,0,2],
            ], columns=['age','sex','cp','trestbps','chol','fbs','restecg',
                        'thalach','exang','oldpeak','slope','ca','thal'])
            csv_template = template_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Tải file CSV mẫu",
                data=csv_template,
                file_name="heart_disease_template.csv",
                mime="text/csv",
            )

        uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                upload_df = pd.read_csv(uploaded_file)

                # Kiểm tra cột
                required_cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
                                  'thalach','exang','oldpeak','slope','ca','thal']
                missing_cols = [c for c in required_cols if c not in upload_df.columns]

                if missing_cols:
                    st.error(f"❌ File thiếu các cột: **{', '.join(missing_cols)}**")
                else:
                    upload_df = upload_df[required_cols]  # đúng thứ tự
                    st.success(f"✅ Đã đọc **{len(upload_df)} bệnh nhân** từ file.")
                    st.dataframe(upload_df, use_container_width=True)

                    csv_model_choice = st.selectbox(
                        "Chọn mô hình dự đoán",
                        list(trained.keys()),
                        index=list(trained.keys()).index(best_name),
                        key="csv_model"
                    )

                    if st.button("🔍 Dự đoán hàng loạt", use_container_width=True):
                        X_upload = scaler.transform(upload_df[required_cols])
                        model_csv = trained[csv_model_choice]['model']
                        probs = model_csv.predict_proba(X_upload)[:, 1]
                        preds = model_csv.predict(X_upload)

                        result_df = upload_df.copy()
                        result_df.insert(0, 'STT', range(1, len(result_df)+1))
                        result_df['Xác suất có bệnh (%)'] = (probs * 100).round(1)
                        result_df['Dự đoán'] = ['🔴 Có nguy cơ' if p==1 else '🟢 Ít nguy cơ' for p in preds]

                        section("📋 Kết quả dự đoán hàng loạt")
                        st.dataframe(
                            result_df[['STT','age','sex','thalach','oldpeak','chol',
                                       'Xác suất có bệnh (%)','Dự đoán']],
                            use_container_width=True
                        )

                        # Biểu đồ phân bố xác suất
                        fig_csv, ax_csv = plt.subplots(figsize=(8, 3))
                        ax_csv.hist(probs * 100, bins=20, color=BLUE, edgecolor='white', alpha=0.8)
                        ax_csv.axvline(50, color=RED, linestyle='--', linewidth=1.5, label='Ngưỡng 50%')
                        ax_csv.set_xlabel('Xác suất có bệnh (%)')
                        ax_csv.set_ylabel('Số bệnh nhân')
                        ax_csv.set_title('Phân bố xác suất nguy cơ', fontweight='bold')
                        ax_csv.legend(); ax_csv.grid(alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_csv); plt.close()

                        # Nút tải kết quả về
                        csv_out = result_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="⬇️ Tải kết quả CSV",
                            data=csv_out,
                            file_name="heart_disease_results.csv",
                            mime="text/csv",
                        )

            except Exception as e:
                st.error(f"❌ Lỗi đọc file: {e}")

    # ── TAB 1: NHẬP THỦ CÔNG ──────────────────────────────────────
    with input_tab1:
        with st.form("predict_form"):
            st.markdown("### 🔢 Thông tin cơ bản")
            c1, c2, c3 = st.columns(3)
            age      = c1.number_input("Tuổi", min_value=20, max_value=100, value=55)
            sex      = c2.selectbox("Giới tính", options=[0, 1], format_func=lambda x: "Nữ" if x==0 else "Nam")
            trestbps = c3.number_input("Huyết áp nghỉ (mmHg)", min_value=80, max_value=220, value=130)

            st.markdown("### ❤️ Chỉ số tim mạch")
            c4, c5, c6 = st.columns(3)
            chol    = c4.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
            thalach = c5.number_input("Nhịp tim tối đa (bpm)", min_value=60, max_value=220, value=150)
            oldpeak = c6.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)

            st.markdown("### 📋 Thông số lâm sàng")
            c7, c8, c9 = st.columns(3)
            cp      = c7.selectbox("Loại đau ngực (cp)", options=[0,1,2,3],
                                   format_func=lambda x: {0:'Không đau',1:'Điển hình',
                                                           2:'Không điển hình',3:'Không TC'}[x])
            fbs     = c8.selectbox("Đường huyết > 120 mg/dl?", options=[0,1],
                                   format_func=lambda x: "Không" if x==0 else "Có")
            restecg = c9.selectbox("Kết quả ECG (restecg)", options=[0,1,2],
                                   format_func=lambda x: {0:'Bình thường',1:'ST bất thường',2:'Phì đại'}[x])

            c10, c11, c12 = st.columns(3)
            exang = c10.selectbox("Đau ngực khi gắng sức?", options=[0,1],
                                  format_func=lambda x: "Không" if x==0 else "Có")
            slope = c11.selectbox("Độ dốc ST (slope)", options=[0,1,2],
                                  format_func=lambda x: {0:'Dốc lên',1:'Phẳng',2:'Dốc xuống'}[x])
            ca    = c12.selectbox("Số mạch chính (ca)", options=[0,1,2,3])

            thal  = st.selectbox("Thalassemia (thal)", options=[1,2,3],
                                 format_func=lambda x: {1:'Bình thường',2:'Fixed defect',3:'Reversible'}[x])

            st.markdown("### 🤖 Chọn mô hình dự đoán")
            model_choice = st.selectbox("Model", list(trained.keys()),
                                        index=list(trained.keys()).index(best_name))

            submitted = st.form_submit_button("🔍 Dự đoán ngay", use_container_width=True)

        # if submitted nằm NGOÀI with st.form nhưng TRONG with input_tab1
        if submitted:
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                    thalach, exang, oldpeak, slope, ca, thal]])
            input_scaled = scaler.transform(input_data)

            model = trained[model_choice]['model']
            prob  = model.predict_proba(input_scaled)[0][1]
            pred  = model.predict(input_scaled)[0]

            st.divider()
            st.markdown("## 📊 Kết quả Dự đoán")

            c1, c2 = st.columns([1, 2])
            with c1:
                if pred == 1:
                    st.markdown(f'<div class="risk-high">🔴 NGUY CƠ CAO<br>Có nguy cơ bệnh tim<br><span style="font-size:2rem">{prob:.1%}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low">🟢 NGUY CƠ THẤP<br>Ít nguy cơ bệnh tim<br><span style="font-size:2rem">{prob:.1%}</span></div>', unsafe_allow_html=True)
                st.metric("Model sử dụng", model_choice)
                st.metric("Xác suất có bệnh", f"{prob:.2%}")
                st.metric("Xác suất không bệnh", f"{(1-prob):.2%}")

            with c2:
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_alpha(0)
                ax.set_aspect('equal')
                ax.axis('off')
                theta = np.linspace(np.pi, 0, 100)
                ax.plot(np.cos(theta), np.sin(theta), color='#ddd', linewidth=20, solid_capstyle='round')
                theta_fill = np.linspace(np.pi, np.pi - prob * np.pi, 100)
                fill_color = RED if prob > 0.5 else ('#F5A623' if prob > 0.3 else BLUE)
                ax.plot(np.cos(theta_fill), np.sin(theta_fill),
                        color=fill_color, linewidth=20, solid_capstyle='round')
                ax.text(0, -0.1, f"{prob:.1%}", ha='center', va='center',
                        fontsize=32, fontweight='bold', color=fill_color)
                ax.text(0, -0.4, "Xác suất có bệnh tim", ha='center', va='center',
                        fontsize=12, color='#555')
                ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.6, 1.2)
                st.pyplot(fig); plt.close()

            st.divider()
            section("📈 Chỉ số của bạn so với 2 nhóm trong dataset")
            numeric_cols = ['age','trestbps','chol','thalach','oldpeak']
            input_dict   = dict(zip(feature_cols, input_data[0]))

            fig, axes = plt.subplots(1, 5, figsize=(16, 3))
            for ax, col in zip(axes, numeric_cols):
                g0 = df[df['target']==0][col]
                g1 = df[df['target']==1][col]
                ax.boxplot([g0, g1], patch_artist=True,
                           boxprops=dict(facecolor='white'),
                           medianprops=dict(color='black', linewidth=2))
                ax.scatter([1, 2], [g0.median(), g1.median()], color=[BLUE, RED], s=50, zorder=5)
                user_val = input_dict[col]
                ax.axhline(user_val, color='purple', linewidth=2, linestyle='--', label='Bạn')
                ax.set_xticks([1, 2]); ax.set_xticklabels(['Không\nbệnh','Có\nbệnh'], fontsize=8)
                ax.set_title(col, fontweight='bold', fontsize=10)
                if col == 'oldpeak':
                    ax.legend(fontsize=7)
            plt.suptitle('Chỉ số của bạn (đường tím) so với 2 nhóm dataset',
                         fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            st.warning("⚠️ **Lưu ý**: Kết quả này được tính bằng mô hình học máy huấn luyện trên dataset Cleveland 1988 (303 mẫu). Đây chỉ là công cụ tham khảo — vui lòng tham khảo ý kiến bác sĩ để được chẩn đoán chính xác.")