# 🥗 NutriChef AI — Smart Nutrition & Meal Planning Assistant

> **Final Project: AI-Powered Conversational Assistant with LLMs and Hybrid Retrieval-Augmented Generation (RAG)**  
> Built with **Python**, **Streamlit**, **Groq Cloud API (Qwen 3.8 27B & Llama 3.3 70B)**, and a **Dual-Engine Vector Store (ChromaDB + BM25)**.

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![LLM Engine](https://img.shields.io/badge/LLM-Groq%20Cloud-f55036.svg)](https://groq.com/)
[![Primary Model](https://img.shields.io/badge/Model-Qwen%203.8%2027B-purple.svg)](https://groq.com/)
[![RAG Architecture](https://img.shields.io/badge/RAG-ChromaDB%20%7C%20BM25-green.svg)](#-system-architecture)
[![License](https://img.shields.io/badge/License-MIT-gray.svg)](LICENSE)

---

## 📖 1. Project Overview & Use Case

**NutriChef AI** is an intelligent nutrition consultant and interactive culinary meal planner designed to address a common daily dilemma: *"What can I cook today that is healthy, budget-friendly, and aligned with my personal fitness and health goals?"*

The system enables users to:
- Generate tailored healthy recipes based on available household ingredients.
- Calculate accurate daily caloric & macronutrient needs (BMR & TDEE formulas).
- Receive personalized meal plans adjusted for dietary goals, medical conditions (e.g., diabetes, hypertension), and food allergies.
- Access evidence-based nutritional knowledge via semantic document retrieval (RAG).

---

## 🎨 2. Creative Parameters & Key Features

NutriChef AI exposes an interactive sidebar with several configurable creative parameters:

### A. Persona & Tone of Voice
- 👨‍🍳 **Friendly & Inspiring Chef**: Warm, motivational, encouraging, and packed with practical culinary tips to make healthy food taste delicious.
- 🩺 **Clinical Dietitian**: Formal, evidence-based, data-driven, focusing on glycemic indexes, lipid profiles, and metabolic health.
- 🏋️‍♂️ **Fitness & Bodybuilding Coach**: Energetic, disciplined, results-oriented, focusing on daily protein targets, lean muscle growth, and workout recovery.

### B. Dietary Goal Customization
- **Caloric Deficit (Fat Loss)**: 300–500 kcal reduction below TDEE with high-protein and high-fiber volume.
- **Clean Bulking (Muscle Gain)**: Controlled calorie surplus with nutrient-dense complex carbs and lean protein sources.
- **Low Carbohydrate / Ketogenic**: Strict carb restriction (<50g/day) to encourage metabolic ketosis.
- **Low Sugar (Diabetes-Friendly)**: Low-GI ingredients (<55) designed to prevent blood glucose spikes.
- **Low Sodium (DASH Diet / Hypertension)**: Sodium restriction (<1,500 mg/day) using aromatic herbs and spices.

### C. Automated Allergy & Restriction Filter
Users can define food intolerances (e.g., *peanuts, shellfish, gluten, dairy*). The assistant dynamically filters out restricted ingredients and suggests safe culinary substitutions without compromising nutritional value.

### D. Advanced LLM Configuration (Groq Engine)
- **Model Switcher**: Powered by Groq's high-speed LPU inference with models including:
  - `qwen/qwen3.8-27b` (Default — cutting-edge reasoning and culinary logic).
  - `qwen-2.5-32b`
  - `llama-3.3-70b-versatile`
  - `llama-3.1-8b-instant` (Ultra-low latency streaming).
  - *Custom Model ID input option*.
- **Creativity Slider (Temperature)**: Adjustable from `0.0` (factual, strictly nutritional) to `1.0` (creative culinary brainstorming).

### E. Hybrid RAG (Retrieval-Augmented Generation)
- **Curated Knowledge Base**: Includes structured nutritional tables (Indonesian TKPI & USDA FoodData Central), evidence-based diet handbooks, and healthy recipes.
- **Dynamic File Ingestion**: Users can upload supplementary documents (PDF, TXT, Markdown) directly through the UI, which are immediately parsed and embedded into the vector store.
- **Hybrid Retrieval**: Every question is searched by both a deterministic **BM25** lexical engine and **ChromaDB** semantic search (`all-MiniLM-L6-v2` embeddings); their rankings are merged with **Reciprocal Rank Fusion (RRF)**. If ChromaDB or its embedding model is unavailable (e.g. offline), retrieval transparently runs on BM25 alone and the sidebar shows which engines are active.

### F. Multi-turn Memory & Export
- **Context-Aware Dialogue**: Maintains sliding-window conversational memory to support follow-up adjustments (e.g., *"Replace the chicken breast with tofu and recalculate the calories"*).
- **Export Chat Transcript**: One-click download of the consultation history in clean Markdown format (`.md`).
- **Session Reset**: Instant context clearance button.

---

## 🏗️ 3. System Architecture

```mermaid
flowchart TD
    User([User]) -->|User Query / Chat Input| UI[Streamlit Web Interface]
    Sidebar[Sidebar Controls] -->|Persona, Diet Goal, Allergies, Temperature| SystemPromptBuilder[System Prompt Generator]
    
    subgraph RAG_Pipeline [Hybrid RAG Pipeline]
        KB[(Curated Knowledge Base: Recipes & Nutrition Tables)] --> Splitter[Markdown / PDF Excerpt Splitter]
        Upload[Dynamic User Upload: PDF/TXT] --> Splitter
        Splitter --> BM25[(BM25 Lexical Index)]
        Splitter --> VectorStore[(ChromaDB Vector Store)]
        UI -->|Query| Retriever[Hybrid Retriever: RRF Fusion]
        BM25 --> Retriever
        VectorStore --> Retriever
    end
    
    Retriever -->|Ground-Truth Context| SystemPromptBuilder
    SystemPromptBuilder --> ChatPayload[Chat Payload + Memory Window]
    
    subgraph LLM_Groq [Groq Cloud Inference]
        ChatPayload --> GroqAPI[Qwen 3.8 27B / Llama 3.3 70B]
        GroqAPI -->|Real-time Token Streaming| UI
    end
    
    UI -->|Structured Recipe Card & Nutrition Breakdown| User
```

---

## 📁 4. Project Directory Structure

```text
nutrichef-ai/
├── app.py                      # Main Streamlit application & interactive UI
├── consultation.py             # Consultation: turn history, RAG retrieval, prompt assembly, LLM streaming
├── llm.py                      # Groq adapter for the LLM port
├── knowledge_base.py           # Knowledge base: Documents, Excerpts, hybrid retrieval (BM25 + ChromaDB, RRF), PDF parser
├── prompts.py                  # Persona definitions, clinical diet rules, and prompt templates
├── tests/                      # pytest suite (runs offline with fake LLM & knowledge base)
├── data/
│   └── knowledge_base/         # Curated factual nutrition datasets
│       ├── resep_sehat_nusantara.md    # Healthy recipe catalog with macro breakdowns
│       ├── tabel_nutrisi_bahan.md      # Calorie and macronutrient reference tables
│       └── panduan_diet_terstruktur.md # Clinical diet handbooks & allergy substitutions
├── .env.example                # Environment variable configuration template
├── .gitignore                  # Git ignore rules (protects credentials and cache)
├── requirements.txt            # Python package dependencies
└── README.md                   # Comprehensive project documentation
```

---

## 🚀 5. Installation & Setup Guide

### Prerequisites
- Python **3.10** or newer installed.
- Free API Key from [Groq Cloud Console](https://console.groq.com/keys).

### Step-by-Step Instructions:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<your-username>/nutrichef-ai.git
   cd nutrichef-ai
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   # On Windows (PowerShell):
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # On macOS / Linux:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables (Optional):**
   Copy `.env.example` to `.env` and insert your Groq API key:
   ```bash
   cp .env.example .env
   ```
   *Note: You can also enter the Groq API key directly into the sidebar field inside the web interface without touching the `.env` file.*

5. **Launch the Application:**
   ```bash
   streamlit run app.py
   ```
   The application will automatically open in your default browser at **`http://localhost:8501`**.

6. **Run the Tests (Optional):**
   ```bash
   pip install pytest
   pytest
   ```

---

## 🧪 6. Evaluation & Testing Scenarios

Reviewers and graders can verify the system using these testing scenarios:

| # | Test Scenario | User Input | Expected Assistant Output |
|---|---|---|---|
| **1** | **Recipe Recommendation & Caloric Deficit** | *"I have chicken breast and oyster mushrooms in my fridge. Suggest a dinner recipe under 250 kcal."* | Retrieves the steamed chicken recipe from the RAG knowledge base, formats measured ingredients, instructions, and provides a clear macro breakdown. |
| **2** | **Persona Switching** | Change persona to *Clinical Dietitian* and ask: *"Why is brown rice preferred over white rice?"* | Explains glycemic index (GI), fiber content, and insulin sensitivity using evidence-based scientific terminology. |
| **3** | **Allergy Constraint Validation** | Enter `Shrimp, Peanuts` in the allergy input, then request a high-protein lunch suggestion. | Guaranteed exclusion of shellfish or peanut dressings, with safe alternative substitutions provided. |
| **4** | **Multi-Turn Context Revision** | Follow up with: *"Replace the carbohydrate source with boiled potatoes and recalculate the macros."* | Accurately remembers prior dish context and recalculates caloric and carbohydrate values based on boiled potatoes. |
| **5** | **Dynamic RAG File Ingestion** | Upload a custom PDF or TXT nutrition note in the sidebar and query its contents. | Splits the new document into excerpts, indexes them in both retrieval engines, and cites the newly uploaded file in the response. |


