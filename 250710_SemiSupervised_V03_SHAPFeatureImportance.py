import streamlit as st
import zipfile
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import shap
import tempfile
import os

st.set_page_config(page_title="Welding NOK Detection", layout="wide")
st.title("⚡ Welding NOK Detection Streamlit App")

# --- Step 1: ZIP uploader ---
uploaded_zip = st.file_uploader("Upload ZIP containing the three CSV files", type="zip")

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        temp_dir = tempfile.mkdtemp()
        zip_ref.extractall(temp_dir)

    file_list = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
    dataframes = {}
    for file in file_list:
        df = pd.read_csv(os.path.join(temp_dir, file))
        dataframes[file] = df

    st.success("Files successfully extracted and loaded.")

    suspected_NOK = {
        "RH_250418_000001_All_68P_RH_01,02NG.csv": [1, 2],
        "RH_250421_160001_TypeA_68p_Gap_Bead67_68NG": [67, 68],
        "LH_250424_123949_LH_A_64P_ALL_47NG": [47]
    }

    # Step 2: Column selection and threshold input
    first_df = next(iter(dataframes.values()))
    column = st.sidebar.selectbox("Select filter column for bead segmentation:", first_df.columns)
    threshold = st.sidebar.number_input("Enter threshold for bead segmentation:", value=0.0)

    # Step 3: Bead segmentation
    def segment_beads(df, column, threshold):
        start_indices = []
        end_indices = []
        signal = df[column].to_numpy()
        i = 0
        while i < len(signal):
            if signal[i] > threshold:
                start = i
                while i < len(signal) and signal[i] > threshold:
                    i += 1
                end = i - 1
                start_indices.append(start)
                end_indices.append(end)
            else:
                i += 1
        return list(zip(start_indices, end_indices))

    if st.button("Segment Beads"):
        beads_data = {}
        for filename, df in dataframes.items():
            segments = segment_beads(df, column, threshold)
            beads_data[filename] = {bead_number: df.iloc[start:end + 1] for bead_number, (start, end) in enumerate(segments, start=1)}
        st.session_state.beads_data = beads_data
        st.success("Bead segmentation completed and dataset locked.")

    if "beads_data" in st.session_state:
        beads_data = st.session_state.beads_data

        # Step 4: Heatmap
        max_beads = max(max(beads.keys()) for beads in beads_data.values())
        heatmap_data = pd.DataFrame(0, index=file_list, columns=list(range(1, max_beads + 1)))
        for filename, beads in beads_data.items():
            for bead_number, df_bead in beads.items():
                heatmap_data.loc[filename, bead_number] = len(df_bead)

        st.subheader("📊 Bead Data Count Heatmap")
        fig, ax = plt.subplots(figsize=(18, 6))
        sns.heatmap(heatmap_data, cmap="viridis", annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

        # Step 5/6: Plotly signal overlay per bead
        bead_to_plot = st.sidebar.number_input("Select Bead Number to visualize:", min_value=1, max_value=max_beads, value=1)
        st.subheader(f"📈 Signal Overlay for Bead {bead_to_plot}")
        fig_plotly = go.Figure()

        for filename, beads in beads_data.items():
            if bead_to_plot in beads:
                df_bead = beads[bead_to_plot]
                y = df_bead[column].to_numpy()
                x = np.arange(len(y))
                color = 'red' if (bead_to_plot in suspected_NOK.get(filename, [])) else None
                fig_plotly.add_trace(go.Scatter(y=y, x=x, mode='lines', name=filename, line=dict(color=color)))

        fig_plotly.update_layout(height=600, xaxis_title="Index within Bead", yaxis_title=column)
        st.plotly_chart(fig_plotly, use_container_width=True)

        # Step 8: Model Training Selection
        st.subheader("🧩 Model Training")
        model_choices = st.multiselect(
            "Select models to train:",
            ["Random Forest", "XGBoost", "Logistic Regression"],
            default=["Random Forest", "XGBoost"]
        )

        if st.button("Train Models"):
            # Feature extraction: simple example features
            X, y, groups = [], [], []
            for filename, beads in beads_data.items():
                for bead_number, df_bead in beads.items():
                    signal = df_bead[column].to_numpy()
                    feature_vector = [
                        np.mean(signal), np.std(signal), np.min(signal), np.max(signal), np.median(signal),
                        np.percentile(signal, 25), np.percentile(signal, 75)
                    ]
                    X.append(feature_vector)
                    y.append(1 if (bead_number in suspected_NOK.get(filename, [])) else 0)
                    groups.append(filename)

            X = np.array(X)
            y = np.array(y)
            groups = np.array(groups)

            models = {}
            if "Random Forest" in model_choices:
                models['Random Forest'] = RandomForestClassifier(class_weight='balanced', random_state=42)
            if "XGBoost" in model_choices:
                models['XGBoost'] = XGBClassifier(scale_pos_weight=(sum(y==0)/sum(y==1)), use_label_encoder=False, eval_metric='logloss', random_state=42)
            if "Logistic Regression" in model_choices:
                models['Logistic Regression'] = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

            st.info("Training in progress...")

            gkf = GroupKFold(n_splits=3)
            for model_name, model in models.items():
                preds = cross_val_predict(model, X, y, groups=groups, cv=gkf, method='predict')
                proba = cross_val_predict(model, X, y, groups=groups, cv=gkf, method='predict_proba')[:, 1]

                st.write(f"### {model_name} Results")
                st.text(classification_report(y, preds, target_names=["OK", "NOK"]))

                cm = confusion_matrix(y, preds)
                disp = ConfusionMatrixDisplay(cm, display_labels=["OK", "NOK"])
                fig_cm, ax_cm = plt.subplots()
                disp.plot(ax=ax_cm)
                st.pyplot(fig_cm)

                # SHAP analysis
                model.fit(X, y)
                explainer = shap.Explainer(model, X)
                shap_values = explainer(X)
                st.write(f"#### SHAP Summary for {model_name}")
                fig_shap = shap.plots.beeswarm(shap_values, show=False)
                st.pyplot(fig_shap)

                st.success(f"{model_name} training and SHAP analysis completed.")

st.caption("🚀 Fully completed pipeline for bead-based NOK detection ready for your dataset workflow.")
