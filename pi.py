import os
# Fix for Windows: Prevent KMeans memory leak UserWarning (must be set before imports)
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest

# =====================================================================
# LALUAN 1: LATIH MODEL MENGGUNAKAN DATA SIFAR (IRIS DATASET)
# =====================================================================
@st.cache_resource
def load_and_train_models():
    # Muat turun data Iris asal
    iris = load_iris()
    X_train = iris.data  

    # Latih model K-Means untuk pengecam keluarga (3 kumpulan)
    # Fix: Explicitly define n_init to avoid scikit-learn FutureWarning
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_train)

    # Latih model Isolation Forest untuk pengesan mutasi (5% anomali)
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_train)
    
    return kmeans, iso_forest

kmeans, iso_forest = load_and_train_models()

st.title("🛸 Sistem Kecerdasan Marikh")
st.write("Sistem sedia menerima data baharu!")

# =====================================================================
# LALUAN 2: INPUT DATA BAHARU DARIPADA PENGGUNA
# =====================================================================

st.header("Sila masukkan ciri-ciri fizikal tumbuhan asing yang anda temui:")

# Mengambil input dari pengguna menggunakan antaramuka web
sepal_length = st.number_input("1. Panjang Sepal (cm)", min_value=0.0, value=5.1, step=0.1)
sepal_width  = st.number_input("2. Lebar Sepal (cm)", min_value=0.0, value=3.5, step=0.1)
petal_length = st.number_input("3. Panjang Kelopak (cm)", min_value=0.0, value=1.4, step=0.1)
petal_width  = st.number_input("4. Lebar Kelopak (cm)", min_value=0.0, value=0.2, step=0.1)

if st.button("Analisis Spesimen"):
    # Tukar input menjadi format array 2D yang diterima oleh Scikit-Learn
    data_baharu = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # =====================================================================
    # LALUAN 3: PROSES PREDIKSI (RAMALAN) DATA BAHARU
    # =====================================================================
    
    # 1. Ramal kumpulan/keluarga
    prediksi_keluarga = kmeans.predict(data_baharu)[0]
    
    # 2. Ramal status mutasi (1 = Normal, -1 = Mutasi)
    prediksi_mutasi = iso_forest.predict(data_baharu)[0]
    
    # =====================================================================
    # LALUAN 4: PAPARAN HASIL ANALISIS SIMULASI
    # =====================================================================
    st.markdown("---")
    st.subheader("📊 HASIL ANALISIS SPESIMEN")
    st.info(f"🧬 Spesimen dimasukkan dalam: **KELUARGA ASING ID-{prediksi_keluarga}**")
    
    # Sediakan URL gambar sebagai ilustrasi untuk setiap keluarga
    iris_images = {
        0: "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg", # Ilustrasi Setosa
        1: "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", # Ilustrasi Versicolor
        2: "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"  # Ilustrasi Virginica
    }
    
    # Papar gambar menggunakan st.image
    st.image(iris_images.get(prediksi_keluarga), caption=f"Ilustrasi Fizikal Keluarga ID-{prediksi_keluarga}", use_column_width=True)

    if prediksi_mutasi == 1:
        st.success("✅ Status: Tumbuhan ini NORMAL dan selamat untuk dikaji.")
        
    else:
        st.error("🚨 Status: WARNING! Mutasi berbahaya dikesan pada sampel ini!")
