import streamlit as st
import zipfile
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks, welch
import shap
import tempfile
import os

st.set_page_config(page_title="Welding NOK Feature Exploration", layout="wide")
st.title("⚡ Welding NOK Feature Exploration with SHAP")

with st.sidebar:
    uploaded_zip = st.file_uploader("Upload ZIP containing your CSV files", type="zip")

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

    first_df = next(iter(dataframes.values()))
    with st.sidebar:
        column = st.selectbox("Select filter column for bead segmentation:", first_df.columns)
        threshold = st.number_input("Enter threshold for bead segmentation:", value=0.0)
        segment_button = st.button("Segment Beads")

    def segment_beads(df, column, threshold):
        start_indices, end_indices = [], []
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

    if segment_button:
        beads_data = {}
        for filename, df in dataframes.items():
            segments = segment_beads(df, column, threshold)
            beads_data[filename] = {bead_number: df.iloc[start:end + 1] for bead_number, (start, end) in enumerate(segments, start=1)}
        st.session_state.beads_data = beads_data
        st.session_state.max_beads = max(max(beads.keys()) for beads in beads_data.values())
        st.success("Bead segmentation completed and locked.")

    if "beads_data" in st.session_state:
        beads_data = st.session_state.beads_data
        max_beads = st.session_state.max_beads

        with st.sidebar:
            bead_to_plot = st.number_input("Select Bead Number to visualize:", min_value=1, max_value=max_beads, value=1)
            column_to_plot = st.selectbox("Select Column to Visualize and Analyze:", first_df.columns)

        st.subheader(f"📈 Signal Overlay for Bead {bead_to_plot} - {column_to_plot}")
        fig_plotly = go.Figure()
        for filename, beads in beads_data.items():
            if bead_to_plot in beads:
                y = beads[bead_to_plot][column_to_plot].to_numpy()
                x = np.arange(len(y))
                color = 'red' if bead_to_plot in suspected_NOK.get(filename, []) else None
                fig_plotly.add_trace(go.Scatter(y=y, x=x, mode='lines', name=filename, line=dict(color=color)))
        fig_plotly.update_layout(height=500, xaxis_title="Index", yaxis_title=column_to_plot)
        st.plotly_chart(fig_plotly, use_container_width=True)

        with st.sidebar:
            train_button = st.button("Extract Features, Train Models, and Analyze")

        if train_button:
            st.info("Feature extraction and model training in progress...")
            def extract_features(signal, fs=1000):
                features = {}
                features['mean'] = np.mean(signal)
                features['std'] = np.std(signal)
                features['min'] = np.min(signal)
                features['max'] = np.max(signal)
                features['median'] = np.median(signal)
                features['q25'] = np.percentile(signal, 25)
                features['q75'] = np.percentile(signal, 75)
                features['rms'] = np.sqrt(np.mean(signal**2))
                features['skew'] = skew(signal)
                features['kurtosis'] = kurtosis(signal)
                features['energy'] = np.sum(signal**2)
                features['entropy'] = entropy(np.histogram(signal, bins=20, density=True)[0]+1e-6)
                peaks, _ = find_peaks(signal)
                features['peak_count'] = len(peaks)
                features['valley_count'] = len(find_peaks(-signal)[0])
                features['zero_cross'] = ((signal[:-1] * signal[1:]) < 0).sum()
                slope = np.diff(signal)
                features['mean_slope'] = np.mean(slope)
                features['max_slope'] = np.max(slope)
                features['slope_std'] = np.std(slope)
                f, Pxx = welch(signal, fs=fs)
                features['dom_freq'] = f[np.argmax(Pxx)]
                features['spec_centroid'] = np.sum(f * Pxx) / np.sum(Pxx)
                features['spec_bw'] = np.sqrt(np.sum(((f - features['spec_centroid'])**2) * Pxx) / np.sum(Pxx))
                band_limits = [(0,50), (50,200), (200,500)]
                for idx, (low, high) in enumerate(band_limits):
                    mask = (f >= low) & (f < high)
                    features[f'bandpower_{idx}'] = np.sum(Pxx[mask])
                features['total_power'] = np.sum(Pxx)
                features['auc'] = np.trapz(np.abs(signal))
                return features

            X, y, groups = [], [], []
            for filename, beads in beads_data.items():
                for bead_number, df_bead in beads.items():
                    signal = df_bead[column_to_plot].to_numpy()
                    features = extract_features(signal)
                    X.append(list(features.values()))
                    y.append(1 if bead_number in suspected_NOK.get(filename, []) else 0)
                    groups.append(filename)

            feature_names = list(features.keys())
            X = np.array(X)
            y = np.array(y)

            models = {
                'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
                'XGBoost': XGBClassifier(scale_pos_weight=(sum(y==0)/max(sum(y==1),1)), use_label_encoder=False, eval_metric='logloss', random_state=42)
            }

            gkf = GroupKFold(n_splits=3)
            for model_name, model in models.items():
                preds = cross_val_predict(model, X, y, groups=groups, cv=gkf, method='predict')
                st.subheader(f"📊 {model_name} Results")
                st.text(classification_report(y, preds, target_names=["OK","NOK"]))
                cm = confusion_matrix(y, preds)
                disp = ConfusionMatrixDisplay(cm, display_labels=["OK","NOK"])
                fig_cm, ax_cm = plt.subplots()
                disp.plot(ax=ax_cm)
                st.pyplot(fig_cm)
                model.fit(X, y)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                st.write(f"#### SHAP Feature Importance for {model_name}")
                try:
                    fig_shap, ax_shap = plt.subplots(figsize=(12,8))
                    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
                    st.pyplot(fig_shap)
                except Exception as e:
                    st.warning(f"SHAP could not be displayed: {e}")
            st.success("Feature extraction, model training, and SHAP analysis completed.")
