"""
app.py - NutriChef AI (Asisten Nutrisi Cerdas & Perencana Menu)
Streamlit Application with Groq LLM & Hybrid RAG System
"""

import os
import streamlit as st
from dotenv import load_dotenv
from consultation import Consultation, Profile
from llm import GroqLLM
from rag_pipeline import RAGPipeline
from prompts import PERSONA_PROMPTS, DIET_GOAL_GUIDELINES

# Muat variabel lingkungan dari .env jika ada
load_dotenv()

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="NutriChef AI - Asisten Nutrisi & Perencana Menu",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling CSS ---
st.markdown("""
<style>
    /* Styling Card & Metric */
    .metric-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-persona {
        background-color: #0284c7;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-diet {
        background-color: #16a34a;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }
    .rag-badge {
        background-color: #f59e0b;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    /* Chat message spacing */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --- Inisialisasi Pipeline RAG di Session State ---
@st.cache_resource(show_spinner=False)
def get_rag_pipeline():
    pipeline = RAGPipeline()
    pipeline.load_knowledge_base()
    return pipeline

rag = get_rag_pipeline()

# Inisialisasi Konsultasi (riwayat Turn) per sesi pengguna
if "consultation" not in st.session_state:
    st.session_state.consultation = Consultation()
consultation = st.session_state.consultation

# --- SIDEBAR: Parameter Kreatif & Konfigurasi Sistem ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1498837167922-ddd27525d352?auto=format&fit=crop&w=500&q=80", use_container_width=True)
    st.title("🥗 NutriChef AI")
    st.caption("Asisten Nutrisi, Diet & Perencana Menu Cerdas")
    
    st.divider()
    
    # 1. Konfigurasi API Key Groq
    st.subheader("🔑 Konfigurasi LLM")
    default_api_key = os.getenv("GROQ_API_KEY", "")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=default_api_key,
        type="password",
        help="Dapatkan API Key gratis di https://console.groq.com/keys"
    )
    
    # Pilihan Model Groq
    model_options = [
        "qwen/qwen3.8-27b",
        "qwen-2.5-32b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "Input Model Kustom..."
    ]
    selected_model_option = st.selectbox(
        "Model LLM Groq",
        options=model_options,
        index=0,
        help="qwen/qwen3.8-27b merupakan model reasoning & percakapan terbaru di Groq yang sangat cepat dan akurat."
    )
    if selected_model_option == "Input Model Kustom...":
        model_choice = st.text_input("Ketik Model ID Groq", value="qwen/qwen3.8-27b")
    else:
        model_choice = selected_model_option
    
    # Slider Kreativitas (Temperature)
    temperature = st.slider(
        "Kreativitas Respon (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Nilai rendah = jawaban lebih faktual & presisi gizi. Nilai tinggi = lebih variatif & kreatif dalam kreasi resep."
    )
    
    st.divider()
    
    # 2. Parameter Persona & Profil Gizi Pengguna
    st.subheader("🎯 Persona & Profil Pengguna")
    
    selected_persona = st.selectbox(
        "Persona AI (Gaya Bahasa)",
        options=list(PERSONA_PROMPTS.keys()),
        index=0
    )
    
    selected_diet = st.selectbox(
        "Target Diet / Pola Makan",
        options=list(DIET_GOAL_GUIDELINES.keys()),
        index=0
    )
    
    user_allergies = st.text_input(
        "Alergi & Pantangan Makanan",
        placeholder="Contoh: Udang, Kacang, Gluten, Dairy",
        help="Bot akan otomatis memfilter dan mengganti bahan-bahan pantangan ini."
    )
    
    st.divider()
    
    # 3. Parameter RAG & Upload Dokumen Tambahan
    st.subheader("📚 Basis Pengetahuan (RAG)")
    st.info(f"Engine: **{rag.engine_type}**\nTotal Chunk Terindeks: **{len(rag.chunks)}**")
    
    uploaded_file = st.file_uploader(
        "Unggah File Panduan/Resep (PDF/TXT)",
        type=["pdf", "txt", "md"],
        help="Perkaya basis data gizi secara dinamis dengan dokumen Anda sendiri."
    )
    if uploaded_file is not None:
        if st.button("📥 Proses & Masukkan ke Knowledge Base"):
            with st.spinner("Memproses dan mengindeks dokumen..."):
                count = rag.add_uploaded_file(uploaded_file.name, uploaded_file)
                st.success(f"Berhasil menambahkan {count} potongan informasi dari '{uploaded_file.name}'!")
                
    show_sources = st.checkbox("Tampilkan Kutipan RAG di Jawaban", value=True)
    
    st.divider()
    
    # 4. Aksi Sesi
    col_reset, col_export = st.columns(2)
    with col_reset:
        if st.button("🗑️ Reset Chat", use_container_width=True):
            consultation.reset()
            st.rerun()
    with col_export:
        if consultation.messages:
            st.download_button(
                label="📥 Unduh Chat",
                data=consultation.transcript_markdown(),
                file_name="nutrichef_chat_history.md",
                mime="text/markdown",
                use_container_width=True
            )


# --- HALAMAN UTAMA ---

# Header Status Badges
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("🥗 NutriChef AI: Konsultan Nutrisi & Resep Sehat")
    st.markdown(
        f"Gaya Konsultasi: <span class='badge-persona'>{selected_persona}</span> | "
        f"Target Diet: <span class='badge-diet'>{selected_diet}</span>",
        unsafe_allow_html=True
    )
with col_head2:
    st.markdown(f"<div style='text-align: right; padding-top: 15px;'><span class='rag-badge'>⚡ RAG Active ({len(rag.chunks)} Docs)</span></div>", unsafe_allow_html=True)

st.write("")

# Banner Sambutan jika Belum Ada Pesan
if not consultation.messages:
    with st.container():
        st.markdown("""
        ### 👋 Selamat Datang di NutriChef AI!
        Saya siap membantu Anda menyusun menu makanan lezat, menghitung estimasi kalori dan makronutrien, 
        serta memberikan solusi resep yang disesuaikan dengan bahan di kulkas dan target kesehatan Anda.
        """)
        
        st.markdown("**💡 Coba tanyakan pertanyaan cepat ini:**")
        preset_cols = st.columns(3)
        with preset_cols[0]:
            if st.button("🍗 Resep Ayam Tinggi Protein (<300 kkal)", use_container_width=True):
                st.session_state.preset_prompt = "Berikan rekomendasi olahan ayam tinggi protein dengan kalori di bawah 300 kkal dan tanpa santan."
        with preset_cols[1]:
            if st.button("⚖️ Cara Menghitung Kebutuhan Kalori (TDEE)", use_container_width=True):
                st.session_state.preset_prompt = "Bagaimana rumus menghitung BMR dan TDEE untuk menentukan target defisit kalori saya?"
        with preset_cols[2]:
            if st.button("🍲 Ide Sarapan Sehat Ramah Diabetes", use_container_width=True):
                st.session_state.preset_prompt = "Buatkan rekomendasi sarapan lezat dengan indeks glikemik rendah yang aman untuk gula darah."


def render_rag_sources(rag_context: str):
    """Menampilkan kutipan RAG di bawah balasan jika ada dan pengguna mengaktifkannya."""
    if rag_context and show_sources:
        with st.expander("🔍 Referensi Basis Pengetahuan (RAG Context)"):
            st.markdown(rag_context)


# Render Riwayat Konsultasi
for msg in consultation.messages:
    with st.chat_message(msg.role):
        st.markdown(msg.content)
        render_rag_sources(msg.rag_context)


# Input Pengguna (Chat Input atau Preset Button)
user_prompt = st.chat_input("Tanyakan rekomendasi resep, bahan makanan, atau konsultasi gizi...")
if "preset_prompt" in st.session_state and st.session_state.preset_prompt:
    user_prompt = st.session_state.preset_prompt
    del st.session_state["preset_prompt"]


if user_prompt:
    # 1. Validasi API Key
    if not groq_api_key:
        st.error("⚠️ Silakan masukkan **Groq API Key** Anda di sidebar terlebih dahulu untuk memulai percakapan!")
        st.stop()

    # 2. Tampilkan pertanyaan pengguna (baru tercatat di Konsultasi setelah balasan selesai)
    with st.chat_message("user"):
        st.markdown(user_prompt)

    profile = Profile(persona=selected_persona, diet_goal=selected_diet, allergies=user_allergies)

    # 3. Jalankan Turn: retrieval RAG + streaming balasan LLM Groq
    with st.chat_message("assistant"):
        try:
            llm = GroqLLM(api_key=groq_api_key, model=model_choice, temperature=temperature)
            with st.spinner("Mencari referensi data gizi dan resep relevan..."):
                st.write_stream(consultation.ask(user_prompt, profile, rag, llm))
            render_rag_sources(consultation.messages[-1].rag_context)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses permintaan ke Groq API: {e}")
