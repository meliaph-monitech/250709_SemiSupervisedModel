import streamlit as st
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert, welch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import io

st.set_page_config(layout="wide")
st.title("⚙️ Circular Welding NOK Detection - Complete Workflow")

# Sidebar for consistent control
with st.sidebar:
    uploaded_zip = st.file_uploader("📁 Upload ZIP containing CSV files", type="zip")
    threshold_value = st.number_input("🔹 Segmentation Threshold:", value=0.5)
    segment_button = st.button("🚀 Segment Beads and Analyze")

# Suspected NOK list
suspected_NOK = {
    "RH_250418_000001_All_68P_RH_01,02NG.csv": [1, 2],
    "RH_250421_160001_TypeA_68p_Gap_Bead67_68NG": [67, 68],
    "LH_250424_123949_LH_A_64P_ALL_47NG": [47]
}

# FFT bands for feature extraction
BANDS = [(100, 300), (300, 600), (600, 1200), (1200, 2400)]
UNIFORM_FREQS = np.linspace(0, 2500, 100)

# Feature extraction function
def extract_features(signal, fs=5000):
    envelope = np.abs(hilbert(signal))
    mean_env = np.mean(envelope)
    std_env = np.std(envelope)
    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    band_energies = [np.mean(psd[(freqs >= start) & (freqs < end)]) if np.any((freqs >= start) & (freqs < end)) else 0 for start, end in BANDS]
    interp_fft = np.interp(UNIFORM_FREQS, freqs, psd, left=psd[0], right=psd[-1])
    return [mean_env, std_env] + band_energies + interp_fft.tolist()

if uploaded_zip and segment_button:
    with zipfile.ZipFile(uploaded_zip, 'r') as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        example_df = pd.read_csv(z.open(csv_files[0]))
        selected_column = st.sidebar.selectbox("🔹 Select Filter Column:", example_df.columns)

        bead_lengths = {}
        bead_signals = {}
        features = []
        labels = []
        files = []
        bead_numbers = []
        bead_max_count = 0

        for file in csv_files:
            df = pd.read_csv(z.open(file))
            signal = df[selected_column].to_numpy()
            segments = []
            i = 0
            while i < len(signal):
                if signal[i] > threshold_value:
                    start = i
                    while i < len(signal) and signal[i] > threshold_value:
                        i += 1
                    end = i
                    segments.append(signal[start:end])
                i += 1
            bead_lengths[file] = [len(seg) for seg in segments]
            bead_signals[file] = segments
            bead_max_count = max(bead_max_count, len(segments))

            for idx, seg in enumerate(segments):
                feat = extract_features(seg)
                features.append(feat)
                bead_num = idx + 1
                files.append(file)
                bead_numbers.append(bead_num)
                labels.append("NOK" if file in suspected_NOK and bead_num in suspected_NOK[file] else "OK")

        # Heatmap
        bead_length_df = pd.DataFrame(index=csv_files, columns=[f"Bead {i+1}" for i in range(bead_max_count)])
        for file in csv_files:
            for idx, length in enumerate(bead_lengths[file]):
                bead_length_df.loc[file, f"Bead {idx+1}"] = length

        st.subheader("📊 Bead-by-Bead Sample Count Heatmap")
        fig, ax = plt.subplots(figsize=(min(20, bead_max_count * 0.5), max(5, len(csv_files) * 0.4)))
        sns.heatmap(bead_length_df.fillna(0).astype(float), annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

        # Feature matrix and training
        X = np.array(features)
        y = np.array(labels)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.subheader("🤖 Training RandomForest Classifier")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_scaled, y)
        preds = clf.predict(X_scaled)

        # PCA visualization
        st.subheader("🔍 PCA Visualization of Features")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["File"] = files
        pca_df["Bead"] = bead_numbers
        pca_df["True Label"] = y
        pca_df["Predicted Label"] = preds

        fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Predicted Label", hover_data=["File", "Bead", "True Label"])
        st.plotly_chart(fig_pca, use_container_width=True)

        # NOK candidate review
        st.subheader("📋 NOK Candidate Table")
        nok_candidates = pca_df[(pca_df["Predicted Label"] == "NOK") & (pca_df["True Label"] == "OK")]
        st.dataframe(nok_candidates[["File", "Bead", "PC1", "PC2"]])

        csv_export = nok_candidates.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download NOK Candidates CSV", csv_export, "nok_candidates.csv", mime="text/csv")

        st.success("✅ Full workflow completed. Ready for field validation and iterative refinement.")
