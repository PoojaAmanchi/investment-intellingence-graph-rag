"""
Builds a real eval_set.json by running actual questions through the working
pipeline and capturing the real answer + retrieved contexts — rather than
hand-typing placeholder data.

Usage:
    python -m eval.build_eval_set
"""
import json
from pathlib import Path

from graph_rag.graph import ask

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "eval_set.json"

QUESTIONS = [
    (
        "Which companies compete with NVDA?",
        "relational",
        "NVIDIA's 10-K names competitors including major cloud service providers, "
        "AI model makers, and other companies building AI/data-center computing "
        "infrastructure, some of which design their own competing chips.",
    ),
    (
        "Which companies compete with AVGO?",
        "relational",
        "Broadcom's 10-K names a broad set of competitors including Nvidia, "
        "Texas Instruments, Qorvo, Skyworks, MediaTek, Samsung, NXP Semiconductors, "
        "and others across its semiconductor and infrastructure software segments.",
    ),
    (
        "What risks does AMD mention about competition?",
        "factual",
        "AMD states that competition could decrease demand for its products, and "
        "that competitors — both existing and new entrants — may offer lower-priced "
        "or higher-performing products, with greater marketing, financial, and "
        "manufacturing resources.",
    ),
    (
        "What risks does QCOM mention about supply chain?",
        "factual",
        "Qualcomm's 10-K discusses risks including supplier disputes, capacity "
        "shortages during high demand, geopolitical disruptions to global supply "
        "chains, and delayed access to key materials and manufacturing technology.",
    ),
    (
        "What does MU say about supply chain risks?",
        "factual",
        "Micron's 10-K discusses supply chain risks tied to global semiconductor "
        "manufacturing dependencies, raw material availability, and geopolitical "
        "conditions affecting its production and sourcing.",
    ),
    (
        "What products does TXN compete on?",
        "factual",
        "Texas Instruments competes across analog and embedded processing products, "
        "facing competition on price, product performance, quality, and customer support "
        "from companies including Analog Devices, Microchip Technology, and NXP Semiconductors.",
    ),
]


def build() -> None:
    records = []
    for question, qtype, ground_truth in QUESTIONS:
        print(f"Running: {question}")
        result = ask(question)

        contexts = [f["text"] for f in result.get("graph_facts", [])]
        contexts += [c["text"] for c in result.get("reranked_chunks", [])]
        if not contexts:
            contexts = [c["text"] for c in result.get("vector_chunks", [])]

        answer = result.get("final_answer") or result.get("draft_answer") or ""

        records.append({
            "question": question,
            "ground_truth": ground_truth,
            "contexts": contexts if contexts else ["(no context retrieved)"],
            "answer": answer,
            "question_type": qtype,
        })
        print(f"  -> confidence={result.get('confidence')}, contexts={len(contexts)}, "
              f"answer_len={len(answer)}\n")

    OUTPUT_PATH.write_text(json.dumps(records, indent=2))
    print(f"Wrote {len(records)} real eval records to {OUTPUT_PATH}")


if __name__ == "__main__":
    build()