"""
test_consultation.py - Pengujian modul Consultation lewat interface-nya (LLM & basis pengetahuan palsu)
"""

import pytest

from consultation import Consultation, Profile, Turn

PROFILE = Profile(
    persona="Ahli Gizi Medis (Clinical Dietitian)",
    diet_goal="Rendah Gula (Diabetes-Friendly)",
    allergies="Udang",
)


class FakeLLM:
    """LLM palsu: mengembalikan potongan teks tetap dan mencatat pesan yang dikirim."""

    def __init__(self, chunks=("Halo", ", ", "Sobat!")):
        self.chunks = list(chunks)
        self.calls = []

    def stream(self, messages):
        self.calls.append(messages)
        yield from self.chunks


class FailingLLM:
    """LLM palsu yang gagal di tengah streaming."""

    def stream(self, messages):
        yield "Pepes "
        raise RuntimeError("Groq API error")


class FakeKnowledgeBase:
    def __init__(self, context="[Konteks #1 | Sumber: resep.md | Topik: Pepes Ayam]\nPepes ayam 220 kkal."):
        self.context = context

    def format_context_for_prompt(self, query, top_k=3):
        return self.context


def test_ask_streams_the_llm_reply_in_order():
    consultation = Consultation()
    llm = FakeLLM(chunks=["Pepes ", "ayam ", "kukus."])

    chunks = list(consultation.ask("Resep ayam rendah kalori?", PROFILE, FakeKnowledgeBase(), llm))

    assert chunks == ["Pepes ", "ayam ", "kukus."]


def test_consumed_reply_is_recorded_as_a_turn_with_its_sources():
    consultation = Consultation()
    knowledge_base = FakeKnowledgeBase(context="[Konteks #1]\nPepes ayam 220 kkal.")

    list(consultation.ask("Resep ayam?", PROFILE, knowledge_base, FakeLLM(chunks=["Pepes ", "ayam."])))

    assert consultation.turns == (
        Turn(question="Resep ayam?", reply="Pepes ayam.", sources="[Konteks #1]\nPepes ayam 220 kkal."),
    )


def test_llm_receives_system_prompt_with_profile_and_retrieved_context_then_the_question():
    llm = FakeLLM()
    knowledge_base = FakeKnowledgeBase(context="[Konteks #1]\nOatmeal IG rendah.")

    list(Consultation().ask("Sarapan ramah diabetes?", PROFILE, knowledge_base, llm))

    sent = llm.calls[0]
    system = sent[0]
    assert system["role"] == "system"
    assert "Formal, ilmiah, objektif" in system["content"]  # persona Ahli Gizi Medis
    assert "Indeks Glikemik (GI) < 55" in system["content"]  # panduan Rendah Gula
    assert "**Udang**" in system["content"]  # alergi
    assert "[Konteks #1]\nOatmeal IG rendah." in system["content"]
    assert sent[-1] == {"role": "user", "content": "Sarapan ramah diabetes?"}


def test_llm_receives_only_the_last_three_turns_before_the_new_question():
    consultation = Consultation()
    knowledge_base = FakeKnowledgeBase()
    for i in range(1, 6):
        list(consultation.ask(f"Q{i}", PROFILE, knowledge_base, FakeLLM(chunks=[f"A{i}"])))
    llm = FakeLLM()

    list(consultation.ask("Q6", PROFILE, knowledge_base, llm))

    assert llm.calls[0][1:] == [
        {"role": "user", "content": "Q3"},
        {"role": "assistant", "content": "A3"},
        {"role": "user", "content": "Q4"},
        {"role": "assistant", "content": "A4"},
        {"role": "user", "content": "Q5"},
        {"role": "assistant", "content": "A5"},
        {"role": "user", "content": "Q6"},
    ]


def test_llm_failure_mid_stream_reaches_the_caller_and_records_nothing():
    consultation = Consultation()
    list(consultation.ask("Q1", PROFILE, FakeKnowledgeBase(), FakeLLM(chunks=["A1"])))
    before = consultation.turns

    with pytest.raises(RuntimeError, match="Groq API error"):
        list(consultation.ask("Q2", PROFILE, FakeKnowledgeBase(), FailingLLM()))

    assert consultation.turns == before


def test_abandoned_stream_records_nothing():
    consultation = Consultation()
    stream = consultation.ask("Resep ayam?", PROFILE, FakeKnowledgeBase(), FakeLLM(chunks=["Pepes ", "ayam."]))

    next(stream)
    stream.close()

    assert consultation.turns == ()


@pytest.mark.parametrize("chunks", [[], ["", ""], ["  ", "\n"]])
def test_empty_reply_is_not_a_turn_and_raises(chunks):
    consultation = Consultation()

    with pytest.raises(RuntimeError, match="tidak mengembalikan balasan"):
        list(consultation.ask("Resep ayam?", PROFILE, FakeKnowledgeBase(), FakeLLM(chunks=chunks)))

    assert consultation.turns == ()


def test_reply_without_sources_is_still_recorded():
    consultation = Consultation()
    llm = FakeLLM(chunks=["Tidak ada data."])

    list(consultation.ask("Apa itu xyz?", PROFILE, FakeKnowledgeBase(context=""), llm))

    assert consultation.turns == (Turn(question="Apa itu xyz?", reply="Tidak ada data.", sources=""),)
    assert "BASIS PENGETAHUAN" not in llm.calls[0][0]["content"]


def test_transcript_markdown_lists_each_question_and_reply_under_its_speaker():
    consultation = Consultation()
    list(consultation.ask("Resep ayam?", PROFILE, FakeKnowledgeBase(), FakeLLM(chunks=["Pepes ayam."])))

    assert consultation.transcript_markdown() == (
        "# Transkrip Konsultasi NutriChef AI\n\n"
        "### Pengguna:\nResep ayam?\n\n"
        "### NutriChef AI:\nPepes ayam.\n\n"
    )


def test_reset_empties_the_consultation():
    consultation = Consultation()
    list(consultation.ask("Resep ayam?", PROFILE, FakeKnowledgeBase(), FakeLLM()))

    consultation.reset()

    assert consultation.turns == ()
