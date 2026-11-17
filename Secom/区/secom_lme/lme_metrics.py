# -*- coding: utf-8 -*-
"""LongMemEval metrics used by run_secom_lme.py.

This module is intentionally standalone: it can be imported by the runner or
executed on an existing predictions.jsonl file.
"""

import argparse
import json
import math
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence


def record_usage(source: str, model: str, completion: Any) -> None:
    path = os.environ.get("SECOM_USAGE_PATH")
    if not path:
        return
    usage = getattr(completion, "usage", None)
    if usage is None:
        return
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif not isinstance(usage, dict):
        usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    row = {"ts": time.time(), "source": source, "model": model, "usage": usage}
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# QA correctness judge (same prompts as the official LongMemEval evaluator)
# ---------------------------------------------------------------------------

def get_anscheck_prompt(
    question_type: str,
    question: str,
    answer: str,
    response: str,
    abstention: bool = False,
) -> str:
    if abstention:
        template = (
            "I will give you an unanswerable question, an explanation, and a "
            "response from a model. Please answer yes if the model correctly "
            "identifies the question as unanswerable. The model could say that "
            "the information is incomplete, or some other information is given "
            "but the asked information is not.\n\n"
            "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
            "Does the model correctly identify the question as unanswerable? "
            "Answer yes or no only."
        )
        return template.format(question, answer, response)

    if question_type in ("single-session-user", "single-session-assistant", "multi-session"):
        template = (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to the "
            "correct answer or contains all the intermediate steps to get the "
            "correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, "
            "answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
            "Model Response: {}\n\nIs the model response correct? "
            "Answer yes or no only."
        )
    elif question_type == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to the "
            "correct answer or contains all the intermediate steps to get the "
            "correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, "
            "answer no. In addition, do not penalize off-by-one errors for the "
            "number of days. If the question asks for the number of "
            "days/weeks/months, etc., and the model makes off-by-one errors "
            "(e.g., predicting 19 days when the answer is 18), the model's "
            "response is still correct. \n\nQuestion: {}\n\nCorrect Answer: "
            "{}\n\nModel Response: {}\n\nIs the model response correct? "
            "Answer yes or no only."
        )
    elif question_type == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response contains some "
            "previous information along with an updated answer, the response "
            "should be considered as correct as long as the updated answer is "
            "the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
            "Model Response: {}\n\nIs the model response correct? "
            "Answer yes or no only."
        )
    elif question_type == "single-session-preference":
        template = (
            "I will give you a question, a rubric for desired personalized "
            "response, and a response from a model. Please answer yes if the "
            "response satisfies the desired response. Otherwise, answer no. "
            "The model does not need to reflect all the points in the rubric. "
            "The response is correct as long as it recalls and utilizes the "
            "user's personal information correctly.\n\nQuestion: {}\n\n"
            "Rubric: {}\n\nModel Response: {}\n\nIs the model response "
            "correct? Answer yes or no only."
        )
    else:
        # Be permissive for future question types instead of crashing a run.
        template = (
            "I will give you a question, a correct answer, and a response from "
            "a model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no.\n\nQuestion: {}\n\nCorrect Answer: "
            "{}\n\nModel Response: {}\n\nIs the model response correct? "
            "Answer yes or no only."
        )

    return template.format(question, answer, response)


def _chat_kwargs(model: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if "deepseek" in model.lower():
        kwargs["extra_body"] = {
            "thinking": {"type": "disabled"},
            "reasoning_effort": "none",
        }
    return kwargs


def judge_qa(
    client: Any,
    model: str,
    question: str,
    answer: str,
    hypothesis: str,
    question_type: str,
    abstention: bool = False,
    max_retries: int = 3,
) -> Optional[bool]:
    prompt = get_anscheck_prompt(
        question_type=question_type,
        question=question,
        answer=answer,
        response=hypothesis,
        abstention=abstention,
    )
    last_err: Any = None
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 10,
            }
            kwargs.update(_chat_kwargs(model))
            completion = client.chat.completions.create(**kwargs)
            record_usage("judge", model, completion)
            text = (completion.choices[0].message.content or "").strip().lower()
            if "yes" in text:
                return True
            if "no" in text:
                return False
            last_err = "unexpected judge output: {!r}".format(text)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(2 ** attempt)
    print("[judge] failed: {}".format(last_err), flush=True)
    return None


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def _dedupe(items: Iterable[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = str(item)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _dcg(relevances: Sequence[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def retrieval_scores(
    retrieved_session_ids: Sequence[Any],
    gold_session_ids: Sequence[Any],
    ks: Sequence[int] = (5, 10),
) -> Dict[str, float]:
    # LongMemEval's retrieval corpus uses a "noans_" prefix for sessions that
    # are not evidence; normalize it so recall is not silently zero.
    def norm(value: Any) -> str:
        return str(value).replace("noans_", "answer_")

    gold = set(norm(x) for x in gold_session_ids)
    retrieved = _dedupe(norm(x) for x in retrieved_session_ids)

    out: Dict[str, float] = {
        "n_gold": float(len(gold)),
        "n_retrieved": float(len(retrieved)),
    }
    for k in ks:
        top = retrieved[:k]
        hits = sum(1 for sid in top if sid in gold)
        out["recall@{}".format(k)] = (hits / len(gold)) if gold else 0.0

        rels = [1 if sid in gold else 0 for sid in top]
        ideal = [1] * min(len(gold), k)
        idcg = _dcg(ideal)
        out["ndcg@{}".format(k)] = (_dcg(rels) / idcg) if idcg else 0.0
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_predictions(
    predictions: Sequence[Dict[str, Any]],
    skip_abs: bool = True,
) -> Dict[str, Any]:
    # Keep the last prediction for a duplicated question id (resume friendly).
    by_qid: Dict[str, Dict[str, Any]] = {}
    for pred in predictions:
        qid = str(pred.get("question_id", ""))
        if qid:
            by_qid[qid] = pred
    rows = list(by_qid.values())

    qa_rows = [p for p in rows if p.get("qa_correct") is not None]
    qa_values = [1.0 if p["qa_correct"] else 0.0 for p in qa_rows]

    qa_by_type: Dict[str, Dict[str, Any]] = {}
    for pred in qa_rows:
        qtype = str(pred.get("question_type") or "unknown")
        bucket = qa_by_type.setdefault(qtype, {"values": []})
        bucket["values"].append(1.0 if pred["qa_correct"] else 0.0)
    qa_by_type = {
        key: {"accuracy": _mean(value["values"]), "n": len(value["values"])}
        for key, value in qa_by_type.items()
    }

    abs_rows = [p for p in qa_rows if p.get("is_abstention")]
    abs_values = [1.0 if p["qa_correct"] else 0.0 for p in abs_rows]

    ret_rows = []
    for pred in rows:
        if skip_abs and pred.get("is_abstention"):
            continue
        gold = pred.get("gold_session_ids") or []
        if not gold:
            continue
        scores = retrieval_scores(
            pred.get("retrieved_session_ids") or [],
            gold,
            ks=(5, 10),
        )
        ret_rows.append((pred, scores))

    ret_by_type: Dict[str, Dict[str, Any]] = {}
    for pred, scores in ret_rows:
        qtype = str(pred.get("question_type") or "unknown")
        bucket = ret_by_type.setdefault(qtype, {"rows": []})
        bucket["rows"].append(scores)
    ret_by_type = {
        key: {
            "n": len(value["rows"]),
            "recall@5": _mean([x["recall@5"] for x in value["rows"]]),
            "ndcg@5": _mean([x["ndcg@5"] for x in value["rows"]]),
            "recall@10": _mean([x["recall@10"] for x in value["rows"]]),
            "ndcg@10": _mean([x["ndcg@10"] for x in value["rows"]]),
        }
        for key, value in ret_by_type.items()
    }

    retrieval = {
        "n": len(ret_rows),
        "recall@5": _mean([x["recall@5"] for _, x in ret_rows]),
        "ndcg@5": _mean([x["ndcg@5"] for _, x in ret_rows]),
        "recall@10": _mean([x["recall@10"] for _, x in ret_rows]),
        "ndcg@10": _mean([x["ndcg@10"] for _, x in ret_rows]),
    }

    return {
        "n_done": len(rows),
        "n_error": sum(1 for p in rows if p.get("error")),
        "n_qa_judged": len(qa_rows),
        "qa_accuracy": _mean(qa_values),
        "qa_by_type": qa_by_type,
        "n_abstention": len(abs_rows),
        "qa_abstention_accuracy": _mean(abs_values),
        "retrieval": retrieval,
        "retrieval_by_type": ret_by_type,
    }


def load_predictions(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LongMemEval predictions.")
    parser.add_argument("--predictions", required=True, help="predictions.jsonl")
    parser.add_argument("--skip-abs", action="store_true", help="skip abstention in retrieval metrics")
    parser.add_argument("--out", default=None, help="optional aggregate.json output path")
    args = parser.parse_args()

    predictions = load_predictions(args.predictions)
    aggregate = aggregate_predictions(predictions, skip_abs=args.skip_abs)
    text = json.dumps(aggregate, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
