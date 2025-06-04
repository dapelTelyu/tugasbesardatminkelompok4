import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import io
import plotly.express as px

st.set_page_config(layout="wide") # Gunakan layout lebar untuk tampilan yang lebih baik

st.title('Dashboard Klasifikasi Produk dan Prediksi Penjualan Harian Maven Roasters Coffee')

# --- Muat Model dan Data ---
st.header("1. Muat Model dan Data")

# Muat model
try:
    kmeans = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler_model.pkl')
    daily_regressor = joblib.load('daily_regressor_model.pkl')
    st.success("Semua model machine learning berhasil dimuat!")
except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan 'kmeans_model.pkl', 'scaler_model.pkl', dan 'daily_regressor_model.pkl' berada di direktori yang sama.")
    st.stop() # Hentikan eksekusi jika model tidak ditemukan

# Muat dataset secara langsung (tidak perlu upload)
try:
    df_raw = pd.read_csv("trimmed_coffee_shop_sales_revenue.csv", delimiter='|')
    st.success("Dataset 'trimmed_coffee_shop_sales_revenue.csv' berhasil dimuat!")
except FileNotFoundError:
    st.error("Dataset 'trimmed_coffee_shop_sales_revenue.csv' tidak ditemukan. Pastikan berada di direktori yang sama.")
    st.stop()

# --- Pra-pemrosesan Data (Ringan) ---
st.header("2. Pra-pemrosesan Data (Ringan)")
df_processed = df_raw.copy()
df_processed['transaction_date'] = pd.to_datetime(df_processed['transaction_date'], format='%Y-%m-%d')
df_processed['transaction_time'] = pd.to_datetime(df_processed['transaction_time'], format='%H:%M:%S').dt.time
df_processed['total_sales'] = df_processed['transaction_qty'] * df_processed['unit_price']

# Terapkan scaler yang sudah dilatih sebelumnya. Pastikan kolom cocok dengan data pelatihan.
numerical_cols_for_scaling = ['transaction_qty', 'unit_price', 'total_sales']
# Periksa apakah kolom-kolom ini ada sebelum scaling
if all(col in df_processed.columns for col in numerical_cols_for_scaling):
    # Pastikan kolom-kolomnya numerik sebelum scaling (penting untuk scaling yang robust)
    for col in numerical_cols_for_scaling:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    # Buat salinan df_processed sebelum scaling untuk inverse transform nanti jika diperlukan
    df_original_scale_for_inverse = df_processed[numerical_cols_for_scaling].copy()

    df_processed[numerical_cols_for_scaling] = scaler.transform(df_processed[numerical_cols_for_scaling])
else:
    st.warning("Tidak dapat melakukan pra-pemrosesan penuh: Kolom numerik yang diperlukan untuk scaling tidak ditemukan.")


st.write("Data telah dipra-proses (konversi tanggal/waktu, perhitungan total_sales, dan normalisasi diterapkan).")
st.write("5 baris pertama data yang diproses:")
st.dataframe(df_processed.head())

# --- Informasi Dataset ---
st.header("3. Informasi Dataset")

st.subheader("Head DataFrame")
st.dataframe(df_raw.head())

st.subheader("Info DataFrame")
buffer = io.StringIO()
df_raw.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.subheader("Deskripsi Data (Kolom Numerik)")
st.dataframe(df_raw.describe())

# --- Hasil K-Means Clustering ---
st.header("4. Hasil K-Means Clustering")

st.markdown(
    """
    K-Means Clustering adalah algoritma pembelajaran tanpa pengawasan yang mengelompokkan
    titik data ke dalam sejumlah klaster berdasarkan kesamaan fitur. Dalam kasus ini, kami telah
    mengelompokkan produk-produk Maven Roasters Coffee berdasarkan
    **Jumlah Transaksi**, **Harga Per Unit**, dan **Total Penjualan** dari setiap transaksi.
    Tujuan dari klasterisasi ini adalah untuk mengidentifikasi segmen-segmen produk yang berbeda
    berdasarkan perilaku penjualan mereka.
    """
)

kmeans_features = ['transaction_qty', 'unit_price', 'total_sales']
if all(col in df_processed.columns for col in kmeans_features):
    df_processed['product_cluster'] = kmeans.predict(df_processed[kmeans_features])

    st.subheader("Pusat Klaster (Centroid setiap klaster - nilai ternormalisasi):")
    st.write("Nilai-nilai ini menunjukkan rata-rata fitur untuk setiap klaster setelah proses normalisasi (skala 0-1).")

    # Membuat DataFrame untuk centroid
    centroid_df = pd.DataFrame(kmeans.cluster_centers_, columns=kmeans_features)
    # Ubah nama kolom indeks (0, 1, 2) menjadi nama yang lebih deskriptif
    centroid_df.index.name = 'ID Klaster'
    st.dataframe(centroid_df.style.format("{:.4f}")) # Format tampilan agar lebih rapi

    st.subheader("Interpretasi Klaster:")
    st.markdown(
        """
        Berdasarkan analisis nilai centroid yang telah dinormalisasi:

        * **Klaster 0: 'Produk Kurang Populer, Penjualan Rendah'**:
            Produk-produk dalam klaster ini memiliki **jumlah transaksi rata-rata yang sangat rendah**,
            **harga per unit rata-rata yang sedang-sedang saja**, dan **total penjualan rata-rata yang juga sangat rendah**.
            Ini mengindikasikan produk yang mungkin kurang diminati atau hanya dibeli sesekali.
        * **Klaster 1: 'Produk Populer, Harga Standar'**:
            Klaster ini berisi produk-produk dengan **jumlah transaksi rata-rata tertinggi**
            namun dengan **harga per unit rata-rata yang relatif standar/menengah**, dan **total penjualan rata-rata yang lebih tinggi dari Klaster 0**.
            Produk-produk ini bisa jadi adalah yang paling sering dibeli dalam jumlah banyak oleh pelanggan.
        * **Klaster 2: 'Produk Premium, Penjualan Tinggi'**:
            Produk di klaster ini memiliki **jumlah transaksi rata-rata yang rendah** (namun tidak serendah Klaster 0),
            **harga per unit rata-rata yang sangat tinggi**, dan menghasilkan **total penjualan rata-rata yang juga sangat tinggi**.
            Ini menunjukkan produk-produk yang mungkin mahal namun bernilai tinggi, dibeli lebih jarang namun memberikan kontribusi besar pada pendapatan per unit.
        """
    )
    st.caption("Catatan: Interpretasi klaster bersifat perkiraan dan dapat disempurnakan dengan analisis lebih lanjut pada kategori/tipe produk asli.")


    st.subheader("Distribusi Produk per Klaster:")
    st.write("Tabel ini menunjukkan berapa banyak produk yang telah ditetapkan ke setiap klaster.")
    cluster_counts = df_processed['product_cluster'].value_counts().reset_index()
    cluster_counts.columns = ['ID Klaster', 'Jumlah Produk'] # Mengubah nama kolom
    st.dataframe(cluster_counts)

    st.subheader("Visualisasi Klaster Produk:")
    st.write("Visualisasi ini menunjukkan bagaimana produk-produk tersebar di setiap klaster berdasarkan fitur 'transaction_qty' dan 'total_sales'.")
    # Menggunakan df_processed yang sudah memiliki kolom 'product_cluster'
    fig_scatter = px.scatter(df_processed,
                             x='transaction_qty',
                             y='total_sales',
                             color='product_cluster', # Warna berdasarkan klaster
                             title='Sebaran Klaster Produk',
                             labels={'transaction_qty': 'Jumlah Transaksi (Normalisasi)',
                                     'total_sales': 'Total Penjualan (Normalisasi)'},
                             hover_name='product_cluster' # Menampilkan ID klaster saat di-hover
                            )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Bagian Baru: Detail Produk per Klaster ---
    st.subheader("Detail Produk per Klaster (Eksplorasi)")
    st.write("Pilih ID Klaster untuk melihat detail produk yang termasuk di dalamnya, serta distribusi kategori dan tipe produknya.")

    selected_cluster_id = st.selectbox('Pilih Klaster ID untuk Detail:', options=sorted(df_processed['product_cluster'].unique()))

    # Filter df_processed berdasarkan klaster yang dipilih
    df_cluster_detail_processed = df_processed[df_processed['product_cluster'] == selected_cluster_id]

    # Gabungkan kembali dengan df_raw untuk mendapatkan detail produk asli (non-normalisasi)
    # Ini penting karena df_processed sudah ternormalisasi, jadi product_category dll akan tetap asli.
    df_display_detail = df_raw.loc[df_cluster_detail_processed.index].copy() # Gunakan .copy() untuk menghindari SettingWithCopyWarning
    df_display_detail['product_cluster'] = df_cluster_detail_processed['product_cluster'] # Tambahkan kembali kolom klaster

    st.write(f"**5 Baris Pertama Produk dalam Klaster {selected_cluster_id}:**")
    st.dataframe(df_display_detail.head())

    st.write(f"**Distribusi Kategori Produk dalam Klaster {selected_cluster_id}:**")
    st.dataframe(df_display_detail['product_category'].value_counts().reset_index().rename(columns={'index': 'Kategori Produk', 'product_category': 'Jumlah'}))

    st.write(f"**Distribusi Tipe Produk dalam Klaster {selected_cluster_id} (5 Teratas):**")
    st.dataframe(df_display_detail['product_type'].value_counts().head(5).reset_index().rename(columns={'index': 'Tipe Produk', 'product_type': 'Jumlah'}))

    st.write(f"**Distribusi Detail Produk dalam Klaster {selected_cluster_id} (5 Teratas):**")
    st.dataframe(df_display_detail['product_detail'].value_counts().head(5).reset_index().rename(columns={'index': 'Detail Produk', 'product_detail': 'Jumlah'}))

else:
    st.warning("Tidak dapat menampilkan hasil K-Means clustering: Kolom yang diperlukan tidak ditemukan dalam data yang diproses.")


# --- Prediksi Penjualan Harian ---
st.header("5. Prediksi Penjualan Harian")
st.write("Prediksi total penjualan untuk hari tertentu dalam seminggu.")

# Opsi untuk hari dalam seminggu (Senin=0, Minggu=6)
day_names = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
day_of_week_input = st.selectbox('Pilih Hari dalam Seminggu:', options=list(range(7)), format_func=lambda x: day_names[x])

if st.button('Prediksi Penjualan Harian'):

    # Pastikan data untuk inverse_transform adalah numpy array dengan bentuk yang benar
    # dan hanya kolom 'total_sales' yang diisi dengan nilai prediksi.
    predicted_daily_sales_normalized = daily_regressor.predict(np.array([[day_of_week_input]]))[0]

    # Buat array dummy dengan nol, lalu masukkan nilai prediksi pada indeks kolom 'total_sales'
    # Ini memastikan bahwa inverse_transform hanya membalikkan skala kolom yang relevan.
    dummy_inverse_transform_array_daily = np.zeros((1, len(numerical_cols_for_scaling)))

    # Dapatkan indeks kolom 'total_sales'
    total_sales_col_index = numerical_cols_for_scaling.index('total_sales')
    dummy_inverse_transform_array_daily[0, total_sales_col_index] = predicted_daily_sales_normalized

    # Lakukan inverse_transform dan ambil nilai 'total_sales' yang sudah dikembalikan skalanya
    predicted_daily_sales_original_scale = scaler.inverse_transform(dummy_inverse_transform_array_daily)[0, total_sales_col_index]

    st.success(f'Prediksi Total Penjualan Harian untuk hari {day_names[day_of_week_input]} adalah: ${predicted_daily_sales_original_scale:.2f}')
