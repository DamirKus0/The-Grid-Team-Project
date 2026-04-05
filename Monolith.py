import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import lightgbm as lgb

RANDOM_STATE = 42
print("✅ Импорты загружены")


# ── Укажи путь к данным ──────────────────────────────────────
TRAIN_PATH = "/content/train_users"   # путь без расширения
TEST_PATH  = "/content/test_users"     # путь без расширения

def load_all(base):
    users  = pd.read_csv(f"{base}.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    props  = pd.read_csv(f"{base}_properties.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    quiz   = pd.read_csv(f"{base}_quizzes.csv", low_memory=False).drop(columns=["Unnamed: 0"], errors="ignore")
    purch  = pd.read_csv(f"{base}_purchases.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    trans  = pd.read_csv(f"{base}_transaction_attempts_v1.csv", low_memory=False).drop(columns=["Unnamed: 0"], errors="ignore")
    try:
        gens = pd.read_csv(f"{base}_generations.csv", low_memory=False).drop(columns=["Unnamed: 0"], errors="ignore")
    except FileNotFoundError:
        gens = None
    return users, props, quiz, purch, trans, gens

tr_users, tr_props, tr_quiz, tr_purch, tr_trans, tr_gens = load_all(TRAIN_PATH)
te_users, te_props, te_quiz, te_purch, te_trans, te_gens = load_all(TEST_PATH)

print(f"Train users: {tr_users.shape}")
print(f"Test users:  {te_users.shape}")
print(f"\nРаспределение классов:")
print(tr_users["churn_status"].value_counts())


tr_users = pd.read_csv('название_вашего_файла.csv')

## 3. EDA — Быстрый анализ

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Распределение классов
tr_users["churn_status"].value_counts().plot(
    kind="bar", ax=axes[0], color=["#2ecc71","#e74c3c","#e67e22"]
)
axes[0].set_title("Churn Distribution")
axes[0].tick_params(axis='x', rotation=15)

# Планы подписки
tr_props["subscription_plan"].value_counts().plot(
    kind="bar", ax=axes[1], color="#3498db"
)
axes[1].set_title("Subscription Plans")
axes[1].tick_params(axis='x', rotation=15)

# Failure codes
tr_trans["failure_code"].value_counts().head(6).plot(
    kind="bar", ax=axes[2], color="#e74c3c"
)
axes[2].set_title("Top Failure Codes")
axes[2].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig("eda.png", dpi=150, bbox_inches="tight")
plt.show()

## 4. Feature Engineering

def build_features(users, props, quiz, purch, trans, gens=None):
    df = users.copy()

    # ── Properties ──────────────────────────────────────────
    props = props.copy()
    props["subscription_start_date"] = pd.to_datetime(props["subscription_start_date"], utc=True, errors="coerce")
    props["sub_start_month"] = props["subscription_start_date"].dt.month
    plan_map = {"Higgsfield Basic":0, "Higgsfield Pro":1, "Higgsfield Creator":2, "Higgsfield Ultimate":3}
    props["plan_tier"] = props["subscription_plan"].map(plan_map).fillna(0).astype(int)
    tier1 = {"US","GB","DE","FR","JP","CA","AU"}
    tier2 = {"IN","BR","KR","TR","MX","ID"}
    props["country_tier"] = props["country_code"].apply(lambda c: 2 if c in tier1 else (1 if c in tier2 else 0))
    props["is_us"] = (props["country_code"] == "US").astype(int)
    props["sub_start_month"] = props["subscription_start_date"].dt.month
    df = df.merge(props[["user_id","plan_tier","country_tier","is_us","sub_start_month"]], on="user_id", how="left")

    # ── Quizzes ──────────────────────────────────────────────
    quiz = quiz.copy()
    quiz["quiz_filled"] = quiz[["source","role","experience","frustration"]].notna().any(axis=1).astype(int)
    for col in ["source","flow_type","frustration","first_feature","role","experience","usage_plan"]:
        if col in quiz.columns:
            le = LabelEncoder()
            quiz[col+"_enc"] = le.fit_transform(quiz[col].fillna("unknown").astype(str))
    quiz_cols = ["user_id","quiz_filled"] + [c for c in quiz.columns if c.endswith("_enc")]
    df = df.merge(quiz[quiz_cols], on="user_id", how="left")

    # ── Purchases ────────────────────────────────────────────
    purch = purch.copy()
    pur_agg = purch.groupby("user_id").agg(
        n_purchases=("transaction_id","count"),
        total_spent=("purchase_amount_dollars","sum"),
        avg_purchase=("purchase_amount_dollars","mean"),
        max_purchase=("purchase_amount_dollars","max"),
        n_purchase_types=("purchase_type","nunique"),
        has_credits=("purchase_type", lambda x: int((x=="Credits package").any())),
        has_upsell=("purchase_type",  lambda x: int((x=="Upsell").any())),
        has_reactivation=("purchase_type", lambda x: int((x=="Reactivation").any())),
    ).reset_index()
    df = df.merge(pur_agg, on="user_id", how="left")

    # ── Transactions ─────────────────────────────────────────
    trans = trans.copy()
    trans["is_failed"] = trans["failure_code"].notna().astype(int)
    trans["is_card_declined"] = (trans["failure_code"] == "card_declined").astype(int)
    tr_agg = trans.groupby("user_id").agg(
        n_transactions=("transaction_id","count"),
        n_failed=("is_failed","sum"),
        n_card_declined=("is_card_declined","sum"),
        total_amount=("amount_in_usd","sum"),
        avg_amount=("amount_in_usd","mean"),
        n_unique_failure_codes=("failure_code","nunique"),
        has_3d_secure=("is_3d_secure","max"),
        is_prepaid=("is_prepaid", lambda x: pd.to_numeric(x,errors="coerce").max()),
        is_virtual=("is_virtual", lambda x: pd.to_numeric(x,errors="coerce").max()),
        is_business=("is_business",lambda x: pd.to_numeric(x,errors="coerce").max()),
    ).reset_index()
    tr_agg["failure_rate"]  = tr_agg["n_failed"]        / (tr_agg["n_transactions"] + 1e-5)
    tr_agg["declined_rate"] = tr_agg["n_card_declined"] / (tr_agg["n_transactions"] + 1e-5)
    for c in ["has_3d_secure","is_prepaid","is_virtual","is_business"]:
        tr_agg[c] = tr_agg[c].fillna(0).astype(int)
    for col in ["card_brand","card_funding","digital_wallet","payment_method_type","cvc_check"]:
        if col in trans.columns:
            mode_s = trans.groupby("user_id")[col].agg(lambda x: x.mode()[0] if len(x.mode())>0 else "unknown")
            le = LabelEncoder()
            tr_agg[col+"_enc"] = le.fit_transform(mode_s.reindex(tr_agg["user_id"]).fillna("unknown").astype(str))
    df = df.merge(tr_agg, on="user_id", how="left")

    # ── Generations ──────────────────────────────────────────
    if gens is not None:
        gen_agg = gens.groupby("user_id").agg(
            n_generations=("generation_id","count"),
            n_completed=("status", lambda x: (x=="completed").sum()),
            n_failed_gen=("status", lambda x: (x=="failed").sum()),
            n_nsfw=("status",       lambda x: (x=="nsfw").sum()),
            total_credits_used=("credit_cost","sum"),
            avg_credits_per_gen=("credit_cost","mean"),
            n_gen_types=("generation_type","nunique"),
        ).reset_index()
        gen_agg["completion_rate"] = gen_agg["n_completed"]  / (gen_agg["n_generations"] + 1e-5)
        gen_agg["nsfw_rate"]       = gen_agg["n_nsfw"]       / (gen_agg["n_generations"] + 1e-5)
        gen_agg["fail_rate_gen"]   = gen_agg["n_failed_gen"] / (gen_agg["n_generations"] + 1e-5)
        df = df.merge(gen_agg, on="user_id", how="left")

    # Заполняем пропуски нулями
    fill_cols = [c for c in df.columns if c not in ["user_id","churn_status"]]
    df[fill_cols] = df[fill_cols].fillna(0)
    return df

print("⚙️  Строим фичи для train...")
df_train = build_features(tr_users, tr_props, tr_quiz, tr_purch, tr_trans, tr_gens)
print(f"✅ Train shape: {df_train.shape}")
print(df_train.head(2))

## 5. Подготовка X, y

label_map     = {"not_churned":0, "vol_churn":1, "invol_churn":2}
label_map_inv = {v:k for k,v in label_map.items()}

df_train["target"] = df_train["churn_status"].map(label_map)

FEATURE_COLS = [c for c in df_train.columns if c not in ["user_id","churn_status","target"]]
X = df_train[FEATURE_COLS]
y = df_train["target"]

print(f"X shape: {X.shape}")
print(f"Классы: {y.value_counts().to_dict()}")

## 6. Cross-Validation (5-Fold)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = []

params = dict(
    objective="multiclass", num_class=3, metric="multi_logloss",
    n_estimators=2000, learning_rate=0.03, num_leaves=127,
    min_child_samples=20, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5,
    class_weight="balanced", random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

    preds = model.predict(X_val)
    score = f1_score(y_val, preds, average="weighted")
    cv_scores.append(score)
    print(f"Fold {fold+1}: weighted F1 = {score:.4f}")

print(f"\n✅ CV Mean: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

## 7. Финальная модель (на всех train данных)

final_model = lgb.LGBMClassifier(**params)
final_model.fit(X, y)
print("✅ Модель обучена!")

# Train report
preds_labels = [label_map_inv[p] for p in final_model.predict(X)]
print("\n📊 Classification Report (Train):")
print(classification_report(df_train["churn_status"], preds_labels))

## 8. Confusion Matrix

cm = confusion_matrix(df_train["churn_status"], preds_labels,
                     labels=["not_churned","vol_churn","invol_churn"])
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["not_churned","vol_churn","invol_churn"],
            yticklabels=["not_churned","vol_churn","invol_churn"])
plt.title("Confusion Matrix (Train)")
plt.ylabel("Реальное"); plt.xlabel("Предсказанное")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.show()

## 9. SHAP — Важность признаков

explainer = shap.TreeExplainer(final_model)
sample = X.sample(2000, random_state=RANDOM_STATE)
shap_values = explainer.shap_values(sample)

# Для multiclass shap_values shape: (n_samples, n_features, n_classes)
if isinstance(shap_values, list):
    mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
else:
    mean_shap = np.abs(shap_values).mean(axis=2)

importance = pd.Series(mean_shap.mean(axis=0), index=sample.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
importance.head(20).sort_values().plot(kind="barh", color="#3498db")
plt.title("Top-20 Features (SHAP)")
plt.tight_layout(); plt.savefig("shap_importance.png", dpi=150); plt.show()

print("\nТоп-15 признаков:")
print(importance.head(15).round(4).to_string())

## 10. Анализ причин оттока

key_cols = ["failure_rate","declined_rate","n_failed","n_transactions",
            "n_purchases","total_spent","plan_tier","has_credits","has_upsell"]
key_cols = [c for c in key_cols if c in df_train.columns]

summary = df_train.groupby("churn_status")[key_cols].mean().round(3)
print("📊 Средние значения по классам:")
print(summary.T.to_string())

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15,8))
axes = axes.flatten()
colors = {"not_churned":"#2ecc71","vol_churn":"#e74c3c","invol_churn":"#e67e22"}
plot_cols = ["failure_rate","n_purchases","total_spent","plan_tier","n_transactions","n_failed"]
plot_cols = [c for c in plot_cols if c in df_train.columns]

for i, col in enumerate(plot_cols):
    for status, color in colors.items():
        vals = df_train[df_train["churn_status"]==status][col]
        vals.clip(upper=vals.quantile(0.95)).hist(
            ax=axes[i], alpha=0.6, label=status, color=color, bins=30)
    axes[i].set_title(col)
    axes[i].legend(fontsize=7)

plt.suptitle("Распределения по группам оттока", fontsize=14)
plt.tight_layout(); plt.savefig("churn_insights.png", dpi=150); plt.show()

## 11. Предсказания для Test + Submission

print("⚙️  Строим фичи для test...")
df_test = build_features(te_users, te_props, te_quiz, te_purch, te_trans, te_gens)

# Выравниваем колонки
for col in FEATURE_COLS:
    if col not in df_test.columns:
        df_test[col] = 0
X_test = df_test[FEATURE_COLS]

preds_test = final_model.predict(X_test)
preds_test_labels = [label_map_inv[p] for p in preds_test]

submission = pd.DataFrame({
    "user_id": df_test["user_id"],
    "churn_status": preds_test_labels
})
submission.to_csv("submission.csv", index=False)

print(f"✅ submission.csv сохранён! ({len(submission)} строк)")
print("\nРаспределение предсказаний:")
print(submission["churn_status"].value_counts())
submission.head(10)

files.download("submission.csv")

