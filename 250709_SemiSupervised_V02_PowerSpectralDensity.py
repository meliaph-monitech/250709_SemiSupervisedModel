import streamlit as st
import zipfile
import os
import io
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.signal import hilbert, welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")
st.title("⚙️ Circular Welding NOK Detection (Feature-Based)")

# Sidebar controls
with st.sidebar:
    uploaded_zip = st.file_uploader("Upload ZIP containing CSV files", type="zip")
    selected_threshold = st.number_input("Segmentation Threshold:", value=0.5)
    segment_button = st.button("Segment Beads")

# Constants for FFT band extraction
BANDS = [(100, 300), (300, 600), (600, 1200), (1200, 2400)]
BAND_LABELS = [f"{b[0]}-{b[1]}Hz" for b in BANDS]
UNIFORM_FREQS = np.linspace(0, 2500, 100)

# Suspected NOK beads
suspected_NOK = {
    "RH_250418_000001_All_68P_RH_01,02NG.csv": [1, 2],
    "RH_250421_160001_TypeA_68p_Gap_Bead67_68NG": [67, 68],
    "LH_250424_123949_LH_A_64P_ALL_47NG": [47]
}

# Feature extraction function
def extract_features(signal, fs=5000):
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    mean_env = np.mean(envelope)
    std_env = np.std(envelope)
    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    band_energies = []
    for start, end in BANDS:
        mask = (freqs >= start) & (freqs < end)
        energy = np.mean(psd[mask]) if np.any(mask) else 0
        band_energies.append(energy)
    interp_fft = np.interp(UNIFORM_FREQS, freqs, psd, left=psd[0], right=psd[-1])
    return [mean_env, std_env] + band_energies + interp_fft.tolist()

if uploaded_zip and segment_button:
    with zipfile.ZipFile(uploaded_zip, 'r') as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        example_df = pd.read_csv(z.open(csv_files[0]))
        selected_column = st.sidebar.selectbox("Select Filter Column:", example_df.columns)

        features = []
        labels = []
        files = []
        bead_numbers = []
        for file in csv_files:
            df = pd.read_csv(z.open(file))
            signal = df[selected_column].to_numpy()
            segments = []
            i = 0
            while i < len(signal):
                if signal[i] > selected_threshold:
                    start = i
                    while i < len(signal) and signal[i] > selected_threshold:
                        i += 1
                    end = i
                    segments.append(signal[start:end])
                i += 1
            for idx, seg in enumerate(segments):
                feat = extract_features(seg)
                features.append(feat)
                bead_num = idx + 1
                files.append(file)
                bead_numbers.append(bead_num)
                if file in suspected_NOK and bead_num in suspected_NOK[file]:
                    labels.append("NOK")
                else:
                    labels.append("OK")

        X = np.array(features)
        y = np.array(labels)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.subheader("🔍 PCA Visualization")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["Label"] = y
        pca_df["File"] = files
        pca_df["Bead"] = bead_numbers
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="Label", hover_data=["File", "Bead"])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🤖 Training RandomForest Classifier")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_scaled, y)
        preds = clf.predict(X_scaled)
        pca_df["Prediction"] = preds
        pca_df["Match"] = pca_df["Label"] == pca_df["Prediction"]

        st.write("### 📈 NOK Candidate Ranking")
        nok_candidates = pca_df[(pca_df["Prediction"] == "NOK") & (pca_df["Label"] == "OK")]
        nok_candidates_sorted = nok_candidates.sort_values(by=["PC1", "PC2"])
        st.dataframe(nok_candidates_sorted[["File", "Bead", "PC1", "PC2"]])

        csv_export = nok_candidates_sorted.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download NOK Candidates CSV", csv_export, "nok_candidates.csv", mime="text/csv")

        st.success("✅ Analysis Complete. Use NOK candidates for human review and dataset refinement.")
