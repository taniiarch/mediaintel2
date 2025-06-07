import streamlit as st
import pandas as pd
import plotly.express as px
import requests # Digunakan dalam penanganan exception, meskipun panggilan API utama menggunakan OpenAI SDK
import json # Digunakan dalam penanganan exception
from io import StringIO
from openai import OpenAI
from openai import APIError # Import specific error for better handling

# --- 0. Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Dasbor Intelijen Media Interaktif",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Fungsi Pembantu untuk Memanggil Gemini API via OpenRouter ---
@st.cache_data(show_spinner="Menghasilkan wawasan dengan Gemini AI...")
def get_gemini_insight(prompt_text, model_name="google/gemini-flash-1.5"): # Default model for OpenRouter
    """
    Memanggil Gemini API via OpenRouter untuk mendapatkan wawasan berdasarkan prompt.
    """
    try:
        # Mengambil API key dari Streamlit Secrets
        # Pastikan Anda telah mengatur file .streamlit/secrets.toml di GitHub Anda
        # dengan kunci seperti OPENROUTER_API_KEY = "sk-..."
        openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
        
        # Inisialisasi klien OpenAI dengan base_url untuk OpenRouter
        client = OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        response = client.chat.completions.create(
            model=model_name, # Specify the model from OpenRouter (e.g., "google/gemini-flash-1.5")
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            # max_tokens=200 # Opsional: batasi panjang respons AI
        )
        
        # Ekstrak teks dari respons
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content
        else:
            st.error(f"Struktur respons AI tidak terduga atau konten kosong: {response}")
            return "Tidak dapat menghasilkan wawasan. (Struktur respons tidak terduga)"

    except KeyError:
        st.error("Error: Kunci API OpenRouter tidak ditemukan di Streamlit Secrets. Pastikan OPENROUTER_API_KEY diatur.")
        return "Gagal mengambil wawasan. Kunci API tidak ditemukan."
    except APIError as e:
        # Tangani error spesifik dari OpenAI SDK (misalnya 403, 429, dll.)
        st.error(f"Kesalahan API saat memanggil AI: {e.status_code} - {e.response.text if e.response else 'Tidak ada respons'}")
        return f"Gagal mengambil wawasan. Kesalahan API: {e.status_code if e.response else 'Tidak diketahui'}"
    except requests.exceptions.ConnectionError as e:
        st.error(f"Kesalahan koneksi saat memanggil AI: {e}")
        return "Gagal mengambil wawasan. Kesalahan koneksi."
    except requests.exceptions.Timeout:
        st.error("Waktu habis saat memanggil AI.")
        return "Gagal mengambil wawasan. Waktu habis."
    except Exception as e:
        st.error(f"Kesalahan tak terduga saat memanggil AI: {e}")
        return f"Gagal mengambil wawasan. Kesalahan: {str(e)}"

# --- Fungsi Normalisasi Nama Kolom ---
def normalize_column_name(name):
    """Menormalisasi nama kolom menjadi format huruf kecil tanpa spasi."""
    return name.lower().replace(' ', '').replace('_', '')

# --- Fungsi Pembersihan Data ---
@st.cache_data(show_spinner="Membersihkan data...")
def clean_data(df):
    """
    Membersihkan DataFrame sesuai dengan spesifikasi:
    - Mengonversi 'Date' ke datetime.
    - Mengisi 'Engagements' yang hilang dengan 0.
    - Menormalisasi nama kolom.
    """
    # Buat salinan DataFrame untuk menghindari SettingWithCopyWarning
    df_cleaned = df.copy()

    # Normalisasi semua nama kolom di awal
    df_cleaned.columns = [normalize_column_name(col) for col in df_cleaned.columns]

    required_columns = ['date', 'platform', 'sentiment', 'location', 'engagements', 'mediatype']
    missing_columns = [col for col in required_columns if col not in df_cleaned.columns]

    if missing_columns:
        st.error(f"Kolom yang diperlukan hilang: {', '.join(missing_columns)}. "
                 f"Pastikan CSV Anda memiliki 'Date', 'Platform', 'Sentiment', 'Location', 'Engagements', 'Media Type' kolom.")
        return pd.DataFrame() # Kembalikan DataFrame kosong jika ada kolom yang hilang

    # Konversi 'date' ke datetime, tangani kesalahan
    # errors='coerce' akan mengubah nilai yang tidak valid menjadi NaT (Not a Time)
    df_cleaned['date'] = pd.to_datetime(df_cleaned['date'], errors='coerce')
    # Hapus baris dengan tanggal yang tidak valid setelah konversi
    df_cleaned.dropna(subset=['date'], inplace=True)

    # Isi 'engagements' yang hilang dengan 0 dan konversi ke int
    df_cleaned['engagements'] = pd.to_numeric(df_cleaned['engagements'], errors='coerce').fillna(0).astype(int)

    # Pastikan kolom-kolom lain adalah string dan tangani nilai NaN
    for col in ['platform', 'sentiment', 'location', 'mediatype']:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(str).fillna('Unknown').str.strip()

    return df_cleaned

# --- Fungsi untuk Membuat dan Menampilkan Grafik ---
def create_chart(df, chart_type, x=None, y=None, color=None, title="", labels={}):
    if df.empty:
        st.warning(f"Tidak ada data untuk membuat grafik: {title}")
        return None

    if chart_type == "pie":
        fig = px.pie(df, names=x, values=y, title=title, hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_traces(textposition='inside', textinfo='percent+label')
    elif chart_type == "line":
        fig = px.line(df, x=x, y=y, title=title, labels=labels,
                      color_discrete_sequence=[px.colors.qualitative.Plotly[0]])
        fig.update_traces(mode='lines+markers')
    elif chart_type == "bar":
        fig = px.bar(df, x=x, y=y, title=title, labels=labels,
                     color=color, color_discrete_sequence=px.colors.qualitative.Plotly)
    else:
        return None

    fig.update_layout(
        font_family="Inter",
        title_font_size=20,
        margin={"t": 50, "b": 50, "l": 50, "r": 50},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    return fig

# --- Main Aplikasi ---
st.title("Dasbor Intelijen Media Interaktif")
st.markdown("oleh **Tania Putri Rachmadani**")

st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    /* Warna pink untuk label uploader */
    .stFileUploader label {
        color: #EA8F8D; /* Sesuaikan dengan warna pink-700 di Tailwind */
    }
    /* Gaya tombol */
    .stButton>button {
        background-color: #DB2777; /* pink-600 */
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 9999px; /* rounded-full */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); /* shadow-lg */
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #BE185D; /* pink-700 */
    }
    .stButton>button:focus:not(:active) {
        box-shadow: 0 0 0 3px rgba(236, 72, 153, 0.5); /* ring-pink-500 */
        outline: none;
    }
    /* Warna spinner saat loading AI insight */
    .stSpinner > div {
        color: #EA8F8D; /* pink-600 */
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Unggah Berkas CSV ---
st.subheader("1. Unggah Berkas CSV")
st.markdown("""
    Harap unggah berkas CSV yang berisi kolom-kolom berikut: 
    <code style="background-color: #FCE7F3; padding: 0.25rem 0.5rem; border-radius: 0.375rem; color: #BE185D;">Date</code>, 
    <code style="background-color: #FCE7F3; padding: 0.25rem 0.5rem; border-radius: 0.375rem; color: #BE185D;">Platform</code>, 
    <code style="background-color: #FCE7F3; padding: 0.25rem 0.5rem; border-radius: 0.375rem; color: #BE185D;">Sentiment</code>, 
    <code style="background-color: #FCE7F3; padding: 0.25rem 0.5rem; border-radius: 0.375rem; color: #BE185D;">Location</code>, 
    <code style="background-color: #FCE7F3; padding: 0.25rem 0.5rem; border-radius: 0.375rem; color: #BE185D;">Engagements</code>, 
    <code style="background-color: #FCE7F3; padding: 0.25rem 0.5rem; border-radius: 0.375rem; color: #BE185D;">Media Type</code>.
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Pilih berkas CSV", type=["csv"])

data = pd.DataFrame() # Inisialisasi data kosong

if uploaded_file is not None:
    # Untuk mengembalikan file sebagai string, gunakan StringIO
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    raw_data = pd.read_csv(stringio)

    with st.spinner("Memproses dan membersihkan data..."):
        data = clean_data(raw_data)

    if not data.empty:
        st.success("Berkas CSV berhasil diunggah dan data dibersihkan!")
        st.write("Pratinjau data yang dibersihkan:")
        st.dataframe(data.head())
    else:
        st.error("Terjadi masalah saat membersihkan data. Harap periksa format CSV Anda.")

# --- 2. Pembersihan Data (Catatan) ---
st.subheader("2. Pembersihan Data")
st.markdown("""
    Data telah dibersihkan secara otomatis:
    - Kolom 'Date' dikonversi menjadi objek datetime.
    - Nilai 'Engagements' yang hilang diisi dengan 0.
    - Nama kolom dinormalisasi untuk konsistensi (misalnya, 'Media Type' menjadi 'mediatype').
    - Baris dengan tanggal yang tidak valid telah dihapus.
""")

# --- 3. Grafik Interaktif ---
if not data.empty:
    st.subheader("3. Grafik Interaktif")
    st.markdown("Jelajahi visualisasi interaktif di bawah ini untuk memahami data media Anda.")

    # --- Grafik 1: Rincian Sentimen (Pie Chart) ---
    st.markdown("---")
    st.markdown("### Rincian Sentimen")
    sentiment_counts = data['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    create_chart(sentiment_counts, "pie", x='sentiment', y='count', title='Rincian Sentimen')
    st.markdown("#### 3 Wawasan Teratas:")
    sentiment_insight = get_gemini_insight(f"Berdasarkan distribusi sentimen berikut ({sentiment_counts.to_dict(orient='records')}), apa 3 wawasan teratas untuk strategi konten media?")
    st.markdown(sentiment_insight)

    # --- Grafik 2: Tren Engagement Seiring Waktu (Line Chart) ---
    st.markdown("---")
    st.markdown("### Tren Engagement Seiring Waktu")
    # Agregasi engagement per tanggal
    engagement_by_date = data.groupby(data['date'].dt.date)['engagements'].sum().reset_index()
    engagement_by_date.columns = ['date', 'total_engagements']
    # Urutkan berdasarkan tanggal untuk tren yang benar
    engagement_by_date = engagement_by_date.sort_values('date')
    create_chart(engagement_by_date, "line", x='date', y='total_engagements',
                     title='Tren Engagement Seiring Waktu',
                     labels={'date': 'Tanggal', 'total_engagements': 'Total Engagement'})
    st.markdown("#### 3 Wawasan Teratas:")
    # Buat ringkasan data untuk prompt Gemini
    if len(engagement_by_date) > 5:
        # Sample data to make prompt shorter for very large datasets
        engagement_trend_summary_for_prompt = pd.concat([
            engagement_by_date.head(2),
            engagement_by_date.sample(n=min(1, len(engagement_by_date) - 4), random_state=42), # min 1 if enough data
            engagement_by_date.tail(2)
        ]).drop_duplicates().sort_values('date').to_dict(orient='records')
    else:
        engagement_trend_summary_for_prompt = engagement_by_date.to_dict(orient='records')

    engagement_insight = get_gemini_insight(f"Menganalisis tren engagement seiring waktu berdasarkan data berikut: {engagement_trend_summary_for_prompt}. Berikan 3 wawasan teratas mengenai pola engagement, termasuk puncak, penurunan, atau anomali data yang relevan untuk performa kampanye media.")
    st.markdown(engagement_insight)

    # --- Grafik 3: Engagement Platform (Bar Chart) ---
    st.markdown("---")
    st.markdown("### Engagement Platform")
    platform_engagements = data.groupby('platform')['engagements'].sum().reset_index()
    platform_engagements = platform_engagements.sort_values('engagements', ascending=False)
    create_chart(platform_engagements, "bar", x='platform', y='engagements',
                     title='Engagement Platform',
                     labels={'platform': 'Platform', 'engagements': 'Total Engagement'},
                     color='platform') # Warna berdasarkan platform
    st.markdown("#### 3 Wawasan Teratas:")
    platform_insight = get_gemini_insight(f"Berdasarkan engagement platform berikut ({platform_engagements.to_dict(orient='records')}), apa 3 wawasan teratas tentang kinerja platform untuk strategi distribusi konten?")
    st.markdown(platform_insight)

    # --- Grafik 4: Campuran Tipe Media (Pie Chart) ---
    st.markdown("---")
    st.markdown("### Campuran Tipe Media")
    media_type_counts = data['mediatype'].value_counts().reset_index()
    media_type_counts.columns = ['mediatype', 'count']
    create_chart(media_type_counts, "pie", x='mediatype', y='count', title='Campuran Tipe Media')
    st.markdown("#### 3 Wawasan Teratas:")
    media_type_insight = get_gemini_insight(f"Mengingat distribusi tipe media ({media_type_counts.to_dict(orient='records')}), apa 3 wawasan teratas mengenai format konten yang paling disukai audiens?")
    st.markdown(media_type_insight)

    # --- Grafik 5: 5 Lokasi Teratas berdasarkan Engagement (Bar Chart) ---
    st.markdown("---")
    st.markdown("### 5 Lokasi Teratas berdasarkan Engagement") # Judul lebih spesifik
    # Mengubah ke top 5 locations by ENGAGEMENT, bukan hanya count
    location_engagements = data.groupby('location')['engagements'].sum().reset_index()
    top_5_locations = location_engagements.sort_values('engagements', ascending=False).head(5)

    create_chart(top_5_locations, "bar", x='location', y='engagements', # Ubah y-axis ke engagements
                     title='5 Lokasi Teratas berdasarkan Engagement',
                     labels={'location': 'Lokasi', 'engagements': 'Total Engagement'}, # Ubah label
                     color='location') # Warna berdasarkan lokasi
    st.markdown("#### 3 Wawasan Teratas:")
    location_insight = get_gemini_insight(f"Berdasarkan 5 lokasi teratas berikut dengan total engagement ({top_5_locations.to_dict(orient='records')}), apa 3 wawasan geografis teratas yang relevan untuk penargetan audiens atau produksi konten lokal?")
    st.markdown(location_insight)


# --- Kesimpulan ---
st.markdown("---")
st.subheader("Dasbor Selesai!")
st.markdown("""
    Anda sekarang dapat berinteraksi dengan grafik dan meninjau wawasan yang dihasilkan.
    Unggah berkas CSV baru untuk menganalisis data yang berbeda.

    **Catatan Penting:** Fungsionalitas unduh PDF (yang ada di versi aplikasi web HTML) tidak dapat diterapkan secara langsung di Streamlit
    karena perbedaan arsitektur (Streamlit berjalan di sisi server, sedangkan unduh PDF yang sebelumnya membutuhkan fungsionalitas sisi klien).
    Anda dapat menggunakan fungsionalitas cetak browser untuk menyimpan halaman sebagai PDF jika diperlukan.
""")
