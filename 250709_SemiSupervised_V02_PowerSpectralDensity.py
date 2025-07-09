import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import streamlit as st
from scipy.signal import hilbert
from scipy.fft import fft
from scipy.stats import pearsonr

st.title("Welding NOK Detection - Complete Pipeline")

# Sidebar inputs
with st.sidebar:
    uploaded_zip = st.file_uploader("Upload ZIP containing CSV files", type="zip")
    threshold_value = st.number_input("Threshold for segmentation:", value=0.5)

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        csv_files = [f for f in z.namelist() if f.endswith(".csv")]
        example_df = pd.read_csv(z.open(csv_files[0]))
        selected_column = st.sidebar.selectbox("Select filter column:", example_df.columns)

        bead_lengths = {}
        bead_signals = {}
        suspected_NOK = {
            "RH_250418_000001_All_68P_RH_01,02NG.csv": [1, 2],
            "RH_250421_160001_TypeA_68p_Gap_Bead67_68NG": [67, 68],
            "LH_250424_123949_LH_A_64P_ALL_47NG": [47]
        }

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
                    segments.append((start, end))
                i += 1
            bead_lengths[file] = [end - start for start, end in segments]
            bead_signals[file] = [signal[start:end] for start, end in segments]

        max_beads = max(len(v) for v in bead_lengths.values())
        heatmap_df = pd.DataFrame(index=csv_files, columns=[f"Bead {i+1}" for i in range(max_beads)])
        for file, lengths in bead_lengths.items():
            for idx, length in enumerate(lengths):
                heatmap_df.loc[file, f"Bead {idx+1}"] = length

        st.subheader("Bead Length Heatmap")
        plt.figure(figsize=(min(20, max_beads * 0.5), max(5, len(csv_files) * 0.5)))
        sns.heatmap(heatmap_df.fillna(0).astype(float), annot=True, fmt=".0f", cmap="viridis")
        st.pyplot(plt)

        st.success("Bead segmentation consistency check complete.")

        # Reference Median + Correlation/RMSE Calculation
        st.subheader("Bead Correlation and RMSE NOK Detection")
        scores = []
        for bead_idx in range(max_beads):
            bead_name = f"Bead {bead_idx+1}"
            reference_signals = []
            for file in csv_files:
                suspected = suspected_NOK.get(file, [])
                if (bead_idx+1) not in suspected and bead_idx < len(bead_signals[file]):
                    reference_signals.append(bead_signals[file][bead_idx])
            if not reference_signals:
                continue
            min_length = min(len(s) for s in reference_signals)
            aligned_refs = np.array([s[:min_length] for s in reference_signals])
            median_ref = np.median(aligned_refs, axis=0)
            for file in csv_files:
                if bead_idx < len(bead_signals[file]):
                    signal = bead_signals[file][bead_idx][:min_length]
                    corr = pearsonr(signal, median_ref)[0]
                    rmse = np.sqrt(np.mean((signal - median_ref)**2))
                    scores.append({
                        "File": file,
                        "Bead": bead_idx+1,
                        "Correlation": corr,
                        "RMSE": rmse
                    })
        score_df = pd.DataFrame(scores).sort_values(by=["Correlation", "RMSE"], ascending=[True, False])
        st.dataframe(score_df)

        st.info("Click a row to view bead signal vs. reference for manual inspection.")
        selected = st.selectbox("Select row to inspect:", score_df.index)
        if selected is not None:
            row = score_df.loc[selected]
            file = row["File"]
            bead_idx = int(row["Bead"]) - 1
            reference_signals = []
            for f in csv_files:
                suspected = suspected_NOK.get(f, [])
                if (bead_idx+1) not in suspected and bead_idx < len(bead_signals[f]):
                    reference_signals.append(bead_signals[f][bead_idx])
            min_length = min(len(s) for s in reference_signals)
            aligned_refs = np.array([s[:min_length] for s in reference_signals])
            median_ref = np.median(aligned_refs, axis=0)
            signal = bead_signals[file][bead_idx][:min_length]
            plt.figure(figsize=(10, 4))
            plt.plot(signal, label="Selected Bead")
            plt.plot(median_ref, label="Reference Median", linestyle="--")
            plt.title(f"{file} - Bead {bead_idx+1}")
            plt.legend()
            st.pyplot(plt)

        st.success("Complete pipeline executed with NOK detection ready for review.")
