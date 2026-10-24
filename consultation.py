"""
consultation.py - Konsultasi NutriChef AI: riwayat Turn, retrieval RAG, dan streaming LLM
Modul ini tidak bergantung pada Streamlit, Groq, maupun ChromaDB agar bisa diuji secara langsung.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, List, Protocol, Tuple

from prompts import build_system_prompt

if TYPE_CHECKING:
    from knowledge_base import KnowledgeBase

# Jumlah Turn terakhir yang ikut dikirim ke LLM agar konteks tetap fokus & efisien
HISTORY_TURNS = 3


class LLM(Protocol):
    """Port LLM: menerima pesan chat dan mengembalikan potongan teks balasan secara streaming."""

    def stream(self, messages: List[Dict[str, str]]) -> Iterator[str]: ...


@dataclass(frozen=True)
class Profile:
    """Persona, target diet, dan alergi yang dipilih pengguna untuk Konsultasi."""
    persona: str
    diet_goal: str
    allergies: str = ""


@dataclass(frozen=True)
class Turn:
    """Satu pertanyaan pengguna beserta balasan NutriChef AI dan Sources (kutipan basis pengetahuan) balasan tersebut."""
    question: str
    reply: str
    sources: str = ""


class Consultation:
    """
    Konsultasi antara satu pengguna dan NutriChef AI.
    Hanya menyimpan Turn yang lengkap: pertanyaan dan balasannya dicatat bersamaan
    setelah streaming selesai, sehingga tidak ada pertanyaan tanpa balasan atau dengan balasan kosong.
    """
    def __init__(self):
        self._turns: List[Turn] = []

    @property
    def turns(self) -> Tuple[Turn, ...]:
        return tuple(self._turns)

    def ask(self, question: str, profile: Profile, knowledge_base: "KnowledgeBase", llm: LLM) -> Iterator[str]:
        """
        Menjalankan satu Turn: retrieval RAG, menyusun system prompt, lalu streaming balasan LLM.
        Turn baru tercatat setelah generator habis dikonsumsi; jika LLM gagal, balasannya kosong,
        atau streaming dihentikan di tengah jalan, tidak ada yang dicatat dan error diteruskan ke pemanggil.
        """
        sources = knowledge_base.find_sources(question)

        system_prompt = build_system_prompt(
            persona_name=profile.persona,
            diet_goal=profile.diet_goal,
            allergies=profile.allergies,
            retrieved_context=sources,
        )
        chat_payload = [{"role": "system", "content": system_prompt}]
        for turn in self._turns[-HISTORY_TURNS:]:
            chat_payload.append({"role": "user", "content": turn.question})
            chat_payload.append({"role": "assistant", "content": turn.reply})
        chat_payload.append({"role": "user", "content": question})

        reply = ""
        for chunk in llm.stream(chat_payload):
            reply += chunk
            yield chunk

        # Balasan kosong bukan balasan: Turn tidak dicatat
        if not reply.strip():
            raise RuntimeError("NutriChef AI tidak mengembalikan balasan.")
        self._turns.append(Turn(question=question, reply=reply, sources=sources))

    def reset(self):
        self._turns = []

    def transcript_markdown(self) -> str:
        """Menyusun transkrip Konsultasi dalam format Markdown untuk diunduh."""
        transcript = "# Transkrip Konsultasi NutriChef AI\n\n"
        for turn in self._turns:
            transcript += f"### Pengguna:\n{turn.question}\n\n"
            transcript += f"### NutriChef AI:\n{turn.reply}\n\n"
        return transcript
