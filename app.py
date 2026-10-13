"""
app.py - NutriChef AI (Asisten Nutrisi Cerdas & Perencana Menu)
Streamlit Application with Groq LLM & Hybrid RAG System
"""

import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from rag_pipeline import RAGPipeline
from prompts import build_system_prompt, PERSONA_PROMPTS, DIET_GOAL_GUIDELINES

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

# Inisialisasi Riwayat Percakapan
if "messages" not in st.session_state:
    st.session_state.messages = []

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
            st.session_state.messages = []
            st.rerun()
    with col_export:
        if st.session_state.messages:
            chat_export_text = "# Transkrip Konsultasi NutriChef AI\n\n"
            for m in st.session_state.messages:
                role_label = "Pengguna" if m["role"] == "user" else "NutriChef AI"
                chat_export_text += f"### {role_label}:\n{m['content']}\n\n"
            st.download_button(
                label="📥 Unduh Chat",
                data=chat_export_text,
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
if not st.session_state.messages:
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


# Render Riwayat Percakapan
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "rag_context" in msg and msg["rag_context"] and show_sources:
            with st.expander("🔍 Referensi Basis Pengetahuan (RAG Context)"):
                st.markdown(msg["rag_context"])


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

    # 2. Tampilkan pesan user ke UI dan simpan ke memory
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 3. Jalankan Retrieval RAG
    with st.spinner("Mencari referensi data gizi dan resep relevan..."):
        retrieved_context_str = rag.format_context_for_prompt(user_prompt, top_k=3)

    # 4. Bangun System Prompt sesuai konfigurasi parameter kreatif saat ini
    system_prompt = build_system_prompt(
        persona_name=selected_persona,
        diet_goal=selected_diet,
        allergies=user_allergies,
        retrieved_context=retrieved_context_str
    )

    # 5. Susun Payload Chat History untuk Multi-turn Context Memory
    chat_payload = [{"role": "system", "content": system_prompt}]
    # Masukkan riwayat pesan sebelumnya (maksimal 8 putaran terakhir agar konteks tetap fokus & efisien)
    for prev in st.session_state.messages[-8:]:
        chat_payload.append({"role": prev["role"], "content": prev["content"]})

    # 6. Panggil LLM Groq dengan Streaming
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            client = Groq(api_key=groq_api_key)
            completion = client.chat.completions.create(
                model=model_choice,
                messages=chat_payload,
                temperature=temperature,
                max_tokens=2048,
                stream=True
            )
            
            for chunk in completion:
                token = chunk.choices[0].delta.content or ""
                full_response += token
                response_placeholder.markdown(full_response + "▌")
                
            response_placeholder.markdown(full_response)
            
            # Tampilkan Expander RAG jika ada data terambil
            if retrieved_context_str and show_sources:
                with st.expander("🔍 Referensi Basis Pengetahuan (RAG Context)"):
                    st.markdown(retrieved_context_str)
                    
            # Simpan respon asisten ke riwayat percakapan
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "rag_context": retrieved_context_str if show_sources else ""
            })
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses permintaan ke Groq API: {e}")
