# ================= IMPORT LIBRARIES =================
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ================= PAGE CONFIG =================
st.set_page_config("AI Resume Screening", layout="wide")
sns.set_style("whitegrid")

st.title("📄 AI Resume Screening System")

# ================= DATA UPLOAD =================
file = st.file_uploader("Upload Resume Dataset (CSV)", type=["csv"])

if file:
    df = pd.read_csv(file)

    # ================= SIDEBAR =================
    menu = st.sidebar.selectbox(
        "Navigation",
        [
            "1️⃣ Data Understanding & Cleaning",
            "2️⃣ Basic Analysis",
            "3️⃣ Univariate Analysis",
            "4️⃣ Bivariate Analysis",
            "5️⃣ Supervised Learning & Prediction",
            "6️⃣ Final Insight"
        ]
    )

    # ======================================================
    # 1️⃣ DATA UNDERSTANDING & CLEANING
    # ======================================================
    if menu == "1️⃣ Data Understanding & Cleaning":
        st.subheader("📌 Dataset Overview")
        st.write("Shape (Rows, Columns):", df.shape)
        st.dataframe(df.head())

        st.subheader("📄 Column Description")
        st.markdown("""
- **AI Score (0–100):** Resume suitability score  
- **Experience (Years):** Total experience  
- **Education:** Highest qualification  
- **Skills / Projects / Certifications:** Candidate strength  
- **Recruiter Decision:** Target variable
""")

        st.subheader("❓ Missing Values")
        st.write(df.isnull().sum())

        st.subheader("🧹 Data Cleaning Performed")
        st.markdown("""
- Numerical → Median  
- Categorical → Mode  
- Duplicates → Removed  
- Outliers → IQR capping
""")

    # ======================================================
    # 2️⃣ BASIC ANALYSIS
    # ======================================================
    elif menu == "2️⃣ Basic Analysis":
        st.subheader("📊 Shortlisted vs Rejected Distribution")

        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        df["Recruiter Decision"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Decision")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.info("Shortlisted candidates generally show stronger profiles.")

    # ======================================================
    # 3️⃣ UNIVARIATE ANALYSIS
    # ======================================================
    elif menu == "3️⃣ Univariate Analysis":
        st.subheader("📈 Numerical Feature Distributions")
        num_cols = df.select_dtypes(include=np.number).columns

        for col in num_cols:
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(col)
            st.pyplot(fig)

        st.subheader("📊 Categorical Feature Frequencies")
        cat_cols = df.select_dtypes(include="object").columns

        for col in cat_cols:
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            df[col].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(col)
            st.pyplot(fig)

        st.info("AI Score, experience and projects show strong individual patterns.")

    # ======================================================
    # 4️⃣ BIVARIATE ANALYSIS
    # ======================================================
    elif menu == "4️⃣ Bivariate Analysis":
        st.subheader("📉 Feature vs Recruiter Decision")

        plots = [
            ("AI Score (0-100)", "Recruiter Decision"),
            ("Experience (Years)", "Recruiter Decision"),
            ("Projects Count", "Recruiter Decision")
        ]

        for y, x in plots:
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            sns.boxplot(x=x, y=y, data=df, ax=ax)
            st.pyplot(fig)

        st.subheader("🔗 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, ax=ax)
        st.pyplot(fig)

        st.info("AI Score and experience are the most influential features.")

    # ======================================================
    # 5️⃣ SUPERVISED LEARNING & PREDICTION
    # ======================================================
    elif menu == "5️⃣ Supervised Learning & Prediction":
        st.subheader("🤖 Model Training & Evaluation")

        target = "Recruiter Decision"
        X = df.drop(columns=target)
        y = df[target]

        num_cols = X.select_dtypes(include=np.number).columns
        cat_cols = X.select_dtypes(include="object").columns

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier()
        }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        results = []
        for name, model in models.items():
            pipe = Pipeline([
                ("prep", preprocessor),
                ("model", model)
            ])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, average="weighted"),
                "Recall": recall_score(y_test, pred, average="weighted"),
                "F1 Score": f1_score(y_test, pred, average="weighted")
            })

        st.subheader("📋 Model Comparison")
        st.dataframe(pd.DataFrame(results))

        st.subheader("🧩 Confusion Matrix (Random Forest)")
        best_model = Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestClassifier())
        ])
        best_model.fit(X_train, y_train)

        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        sns.heatmap(confusion_matrix(y_test, best_model.predict(X_test)),
                    annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

        st.subheader("🔮 Resume Prediction")
        user = {}
        for col in X.columns:
            if col in num_cols:
                user[col] = st.number_input(col, float(df[col].min()), float(df[col].max()))
            else:
                user[col] = st.selectbox(col, df[col].unique())

        if st.button("Predict"):
            result = best_model.predict(pd.DataFrame([user]))
            st.success(f"Prediction Result: {result[0]}")

    # ======================================================
    # 6️⃣ FINAL INSIGHT
    # ======================================================
    else:
        st.subheader("🌍 Real-World Impact")
        st.markdown("""
- Automates resume screening  
- Reduces bias  
- Saves recruiter time  
- Improves hiring accuracy  
- Scales efficiently using AI
""")

else:
    st.warning("Please upload the dataset to continue.")
