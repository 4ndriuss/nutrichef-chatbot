"""
consultation.py - Konsultasi NutriChef AI: riwayat Turn, retrieval RAG, dan streaming LLM
Modul ini tidak bergantung pada Streamlit, Groq, maupun ChromaDB agar bisa diuji secara langsung.
"""

from dataclasses import dataclass
from typing import Dict, Iterator, List, Protocol, Tuple

from prompts import build_system_prompt

# Jumlah Turn terakhir yang ikut dikirim ke LLM agar konteks tetap fokus & efisien
HISTORY_TURNS = 3
# Jumlah chunk basis pengetahuan yang diambil per pertanyaan
RETRIEVAL_TOP_K = 3


class LLM(Protocol):
    """Port LLM: menerima pesan chat dan mengembalikan potongan teks balasan secara streaming."""

    def stream(self, messages: List[Dict[str, str]]) -> Iterator[str]: ...


class KnowledgeBase(Protocol):
    def format_context_for_prompt(self, query: str, top_k: int = 3) -> str: ...


@dataclass(frozen=True)
class Profile:
    """Persona, target diet, dan alergi yang dipilih pengguna untuk Konsultasi."""
    persona: str
    diet_goal: str
    allergies: str = ""


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    rag_context: str = ""


class Consultation:
    """
    Konsultasi antara satu pengguna dan NutriChef AI.
    Hanya menyimpan Turn yang lengkap: pertanyaan dan balasannya dicatat bersamaan
    setelah streaming selesai, sehingga tidak ada pertanyaan tanpa balasan.
    """
    def __init__(self):
        self._messages: List[Message] = []

    @property
    def messages(self) -> Tuple[Message, ...]:
        return tuple(self._messages)

    def ask(self, prompt: str, profile: Profile, knowledge_base: KnowledgeBase, llm: LLM) -> Iterator[str]:
        """
        Menjalankan satu Turn: retrieval RAG, menyusun system prompt, lalu streaming balasan LLM.
        Turn baru tercatat setelah generator habis dikonsumsi; jika LLM gagal atau streaming
        dihentikan di tengah jalan, tidak ada yang dicatat dan error diteruskan ke pemanggil.
        """
        rag_context = knowledge_base.format_context_for_prompt(prompt, top_k=RETRIEVAL_TOP_K)

        system_prompt = build_system_prompt(
            persona_name=profile.persona,
            diet_goal=profile.diet_goal,
            allergies=profile.allergies,
            retrieved_context=rag_context,
        )
        # Setiap Turn terdiri dari 2 pesan (pertanyaan + balasan)
        recent = self._messages[-2 * HISTORY_TURNS:]
        chat_payload = [{"role": "system", "content": system_prompt}]
        chat_payload += [{"role": m.role, "content": m.content} for m in recent]
        chat_payload.append({"role": "user", "content": prompt})

        reply = ""
        for chunk in llm.stream(chat_payload):
            reply += chunk
            yield chunk

        self._messages.append(Message(role="user", content=prompt))
        self._messages.append(Message(role="assistant", content=reply, rag_context=rag_context))

    def reset(self):
        self._messages = []

    def transcript_markdown(self) -> str:
        """Menyusun transkrip Konsultasi dalam format Markdown untuk diunduh."""
        transcript = "# Transkrip Konsultasi NutriChef AI\n\n"
        for m in self._messages:
            role_label = "Pengguna" if m.role == "user" else "NutriChef AI"
            transcript += f"### {role_label}:\n{m.content}\n\n"
        return transcript
