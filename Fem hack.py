#=================== IMPORT LIBRARIES ================================
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#=================== PAGE CONFIG ================================
st.set_page_config(page_title='AI Resume Screening', layout='wide')

#=================== DATASET UPLOAD ================================
st.subheader("📁 Dataset Input")
data_source = st.radio("Choose how to load the dataset:", ("Upload from UI", "Load from Code"))

df = None
if data_source == "Upload from UI":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
elif data_source == "Load from Code":
    file_path = "AI_Resume_Screening.csv"  # change path if needed
    try:
        df = pd.read_csv(file_path)
        st.success("Dataset loaded from code!")
    except FileNotFoundError:
        st.error("File not found. Check the file path.")

#=================== SIDEBAR ================================
if df is not None:
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio(
        "Go to",
        ["UNIVARIATE", "BIVARIATE", "SUPERVISED LEARNING"]
    )

    #=================== UNIVARIATE ================================
    if menu == "UNIVARIATE":
        st.subheader("📊 Part 3: Univariate Analysis")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        if numeric_cols:
            st.subheader("1️⃣ Numerical Feature Distributions")
            for col in numeric_cols:
                st.write(f"**{col} Distribution:**")
                fig = px.histogram(df, x=col, nbins=20, marginal="box", title=f"{col} Distribution")
                st.plotly_chart(fig)
                st.text_area(f"Interpretation for {col}", "Add your interpretation here...")

        if categorical_cols:
            st.subheader("2️⃣ Categorical Feature Frequencies")
            for col in categorical_cols:
                st.write(f"**{col} Frequency:**")
                fig = px.histogram(df, x=col, color=col, barmode='group', title=f"{col} Frequency")
                st.plotly_chart(fig)
                st.text_area(f"Interpretation for {col}", "Add your interpretation here...")

        st.subheader("3️⃣ Overall Insights")
        st.text_area("Overall Insights", "Add your overall insights here...")

    #=================== BIVARIATE ================================
    elif menu == "BIVARIATE":
        st.subheader("📊 Part 4: Bivariate Analysis")
        if 'status' not in df.columns:
            st.warning("Bivariate analysis requires 'status' column (Shortlisted/Rejected).")
        else:
            # 1. AI Score vs Status
            st.subheader("1️⃣ AI Score vs Candidate Status")
            fig = px.box(df, x='status', y='ai_score', color='status', title="AI Score by Status")
            st.plotly_chart(fig)
            st.text_area("Interpretation for AI Score vs Status", "Add your interpretation here...")

            # 2. Experience vs Status
            if 'experience' in df.columns:
                st.subheader("2️⃣ Experience vs Status")
                fig = px.box(df, x='status', y='experience', color='status', title="Experience by Status")
                st.plotly_chart(fig)
                st.text_area("Interpretation for Experience vs Status", "Add your interpretation here...")

            # 3. Education vs Status
            if 'education' in df.columns:
                st.subheader("3️⃣ Education Level vs Status")
                fig = px.histogram(df, x='education', color='status', barmode='group', title="Education vs Status")
                st.plotly_chart(fig)
                st.text_area("Interpretation for Education vs Status", "Add your interpretation here...")

            # 4. Skills vs Status
            if 'num_skills' in df.columns:
                st.subheader("4️⃣ Number of Skills vs Status")
                fig = px.box(df, x='status', y='num_skills', color='status', title="Skills vs Status")
                st.plotly_chart(fig)
                st.text_area("Interpretation for Skills vs Status", "Add your interpretation here...")

            # 5. Projects vs Status
            if 'num_projects' in df.columns:
                st.subheader("5️⃣ Projects vs Status")
                fig = px.box(df, x='status', y='num_projects', color='status', title="Projects vs Status")
                st.plotly_chart(fig)
                st.text_area("Interpretation for Projects vs Status", "Add your interpretation here...")

            # 6. Salary vs Status
            if 'salary_expectation' in df.columns:
                st.subheader("6️⃣ Salary Expectation vs Status")
                fig = px.box(df, x='status', y='salary_expectation', color='status', title="Salary vs Status")
                st.plotly_chart(fig)
                st.text_area("Interpretation for Salary vs Status", "Add your interpretation here...")

            # 7. AI Score across Job Roles
            if 'job_role' in df.columns:
                st.subheader("7️⃣ AI Score across Job Roles")
                fig = px.box(df, x='job_role', y='ai_score', color='job_role', title="AI Score vs Job Role")
                st.plotly_chart(fig)
                st.text_area("Interpretation for AI Score vs Job Role", "Add your interpretation here...")

            # 8. Correlation: Experience vs AI Score
            if 'experience' in df.columns and 'ai_score' in df.columns:
                st.subheader("8️⃣ Correlation: Experience vs AI Score")
                corr = df['experience'].corr(df['ai_score'])
                st.write(f"Correlation coefficient: {corr:.2f}")
                fig = px.scatter(df, x='experience', y='ai_score', color='status', trendline='ols',
                                 title="Experience vs AI Score")
                st.plotly_chart(fig)
                st.text_area("Interpretation for Experience vs AI Score", "Add your interpretation here...")

            # 9. Certifications vs Status
            if 'certifications' in df.columns:
                st.subheader("9️⃣ Certifications vs Status")
                cert_counts = df.groupby(['status','certifications']).size().reset_index(name='count')
                fig = px.bar(cert_counts, x='certifications', y='count', color='status', barmode='group',
                             title="Certifications vs Status")
                st.plotly_chart(fig)
                st.text_area("Interpretation for Certifications vs Status", "Add your interpretation here...")

            # 10. Education vs AI Score
            if 'education' in df.columns and 'ai_score' in df.columns:
                st.subheader("🔟 Education vs AI Score")
                fig = px.box(df, x='education', y='ai_score', color='education', title="Education vs AI Score")
                st.plotly_chart(fig)
                st.text_area("Interpretation for Education vs AI Score", "Add your interpretation here...")

            # 11. Experience vs Job Role
            if 'job_role' in df.columns and 'experience' in df.columns:
                st.subheader("11️⃣ Experience vs Job Roles")
                fig = px.box(df, x='job_role', y='experience', color='job_role', title="Experience vs Job Roles")
                st.plotly_chart(fig)
                st.text_area("Interpretation for Experience vs Job Roles", "Add your interpretation here...")

            # 12. Salary Outliers
            if 'salary_expectation' in df.columns:
                st.subheader("12️⃣ Salary Outliers")
                Q1 = df['salary_expectation'].quantile(0.25)
                Q3 = df['salary_expectation'].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5*IQR
                upper = Q3 + 1.5*IQR
                outliers = df[(df['salary_expectation'] < lower) | (df['salary_expectation'] > upper)]
                st.write(f"Salary outliers count: {len(outliers)}")
                fig = px.box(df, y='salary_expectation', title="Salary Expectation Boxplot")
                st.plotly_chart(fig)
                st.text_area("Interpretation for Salary Outliers", "Add your interpretation here...")

            # 13. Projects vs Status
            if 'num_projects' in df.columns:
                st.subheader("13️⃣ Projects Count vs Status")
                fig = px.histogram(df, x='num_projects', color='status', barmode='group', title="Projects Count vs Status")
                st.plotly_chart(fig)
                st.text_area("Interpretation for Projects Count vs Status", "Add your interpretation here...")

            # 14. Correlation Heatmap
            st.subheader("14️⃣ Numerical Features Correlation")
            numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
            if 'status' in numeric_cols:
                numeric_cols.remove('status')
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap")
            st.plotly_chart(fig)
            st.text_area("Interpretation for Correlation Heatmap", "Add your interpretation here...")

            # 15. Overall Patterns
            st.subheader("15️⃣ Overall Patterns for Shortlisting")
            st.text_area("Insights / Patterns", "Add your insights here...")

    #=================== SUPERVISED LEARNING ================================
    elif menu == "SUPERVISED LEARNING":
        st.subheader("🔹 Supervised Learning")

        target_options = df.columns.tolist()
        target = st.selectbox("Select Target Column", target_options)
        features = st.multiselect("Select Feature Columns", [c for c in df.columns if c != target])
        if not features:
            features = [c for c in df.columns if c != target]

        X = df[features]
        y = df[target]
        numeric_X = X.select_dtypes(include=['int64','float64']).columns.tolist()
        categorical_X = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_X:
            X = pd.get_dummies(X, columns=categorical_X)
        if y.dtype=='object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)

        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True, random_state=42)
        }

        selected_models = st.multiselect("Select Models", list(models.keys()), default=list(models.keys()))
        threshold = st.number_input("Overfitting Threshold (Train-Test Gap)", 0.01, 1.0, 0.10)

        if st.button("Train Models"):
            results = {}
            best_model_name = None
            best_accuracy = 0
            scaler = None

            for name in selected_models:
                model = models[name]
                X_train_scaled, X_test_scaled = X_train.copy(), X_test.copy()

                if name in ["Logistic Regression","KNN","SVM"] and numeric_X:
                    scaler = StandardScaler()
                    X_train_scaled[numeric_X] = scaler.fit_transform(X_train_scaled[numeric_X])
                    X_test_scaled[numeric_X] = scaler.transform(X_test_scaled[numeric_X])

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                train_acc = model.score(X_train_scaled, y_train)
                test_acc = accuracy_score(y_test, y_pred)

                # Overfitting check
                gap = train_acc - test_acc
                if gap > threshold:
                    st.warning(f"{name} overfitting detected (Gap={gap*100:.2f}%)")
                    if name=="Decision Tree":
                        model = DecisionTreeClassifier(max_depth=3,min_samples_leaf=10,random_state=42)
                    elif name=="Random Forest":
                        model = RandomForestClassifier(max_depth=3,min_samples_leaf=10,random_state=42)
                    elif name=="Logistic Regression":
                        model = LogisticRegression(C=0.05,max_iter=500)
                    elif name=="SVM":
                        model = SVC(C=0.5, kernel='linear', probability=True, random_state=42)
                    elif name=="KNN":
                        model = KNeighborsClassifier(n_neighbors=10)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    train_acc = model.score(X_train_scaled, y_train)
                    test_acc = accuracy_score(y_test, y_pred)

                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                results[name] = {"Train Accuracy": train_acc, "Test Accuracy": test_acc,
                                 "Precision": precision, "Recall": recall, "F1 Score": f1,
                                 "Model Object": model}
                if test_acc > best_accuracy:
                    best_accuracy = test_acc
                    best_model_name = name

            perf_df = pd.DataFrame({
                "Train Accuracy":[results[m]["Train Accuracy"] for m in results],
                "Test Accuracy":[results[m]["Test Accuracy"] for m in results],
                "Precision":[results[m]["Precision"] for m in results],
                "Recall":[results[m]["Recall"] for m in results],
                "F1 Score":[results[m]["F1 Score"] for m in results]
            }, index=results.keys())

            st.subheader("📈 Model Performance")
            fig = px.bar(perf_df, barmode='group', title="Metrics Comparison")
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"✅ Best Model: {best_model_name} | Test Accuracy: {best_accuracy:.4f}")
