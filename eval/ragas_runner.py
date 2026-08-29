import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

"""
RAGAS evaluation, using a local Ollama model as the judge via
LangchainLLMWrapper/LangchainEmbeddingsWrapper.

This is the fix for the RAGAS blocker from the FOMC project: that project
used Gemini Flash-Lite, which silently ignores `max_tokens` and
`temperature`, breaking RAGAS's internal calls. Local Ollama models respect
every parameter passed to them, so this compatibility chain doesn't recur.

Before running the full eval set, sanity check the judge connection with
one manual call (see `if __name__` block) — cheaper to catch a connection
problem on one call than after burning through the whole eval set.
"""
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from graph_rag.config import settings

EVAL_SET_PATH = Path(__file__).resolve().parent / "data" / "eval_set.json"


def get_ragas_llm() -> LangchainLLMWrapper:
    judge = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=0)
    return LangchainLLMWrapper(judge)


def get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    embeddings = OllamaEmbeddings(model=settings.ollama_embed_model, base_url=settings.ollama_base_url)
    return LangchainEmbeddingsWrapper(embeddings)


def load_eval_dataset() -> Dataset:
    records = json.loads(EVAL_SET_PATH.read_text())
    return Dataset.from_list(records)


def run_ragas_eval(dataset: Dataset) -> pd.DataFrame:
    # max_workers kept low: a single local Ollama instance can't handle
    # heavy concurrent request volume the way a hosted API can.
    run_config = RunConfig(max_workers=1, timeout=180)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
        run_config=run_config,
    )
    return result.to_pandas()


def run_ragas_eval_from_file() -> pd.DataFrame:
    dataset = load_eval_dataset()
    return run_ragas_eval(dataset)


def run_comparison_by_question_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups RAGAS scores by question_type so you can see whether graph
    retrieval actually outperforms vector-only on relational/temporal
    questions. This comparison table is the strongest artifact for
    your portfolio write-up.
    """
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    available_cols = [c for c in metric_cols if c in df.columns]
    return df.groupby("question_type")[available_cols].mean().round(3)


if __name__ == "__main__":
    print("Sanity-checking Ollama judge connection...")
    test_llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url)
    test_response = test_llm.invoke("Reply with just the word 'ok'.")
    print(f"Judge model responded: {test_response.content!r}")

    print("\nRunning full RAGAS eval...")
    scores_df = run_ragas_eval_from_file()

    import pandas as pd
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(scores_df)

    RESULTS_PATH = Path(__file__).resolve().parent / "data" / "ragas_results.csv"
    scores_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved full results to {RESULTS_PATH}")

    if "question_type" in scores_df.columns:
        print("\nComparison by question type:")
        comparison = run_comparison_by_question_type(scores_df)
        print(comparison)

        print("\nOverall averages across all questions:")
        metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        available_cols = [c for c in metric_cols if c in scores_df.columns]
        print(scores_df[available_cols].mean().round(3))