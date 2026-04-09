from __future__ import annotations

from typing import Any, Optional


def merge_kv_connector_outputs(a: Any, b: Any) -> Any:
    if a is None:
        return b
    if b is None:
        return a
    cls = type(a)
    finished_sending = set(getattr(a, "finished_sending", set()) or set()) | set(
        getattr(b, "finished_sending", set()) or set()
    )
    finished_recving = set(getattr(a, "finished_recving", set()) or set()) | set(
        getattr(b, "finished_recving", set()) or set()
    )
    return cls(
        finished_sending=finished_sending or None,
        finished_recving=finished_recving or None,
    )


def merge_v1_runner_outputs(original_order: list[str], out_a: Any, out_b: Any) -> Any:
    if not hasattr(out_a, "req_ids") or not hasattr(out_b, "req_ids"):
        raise TypeError("v1 split merge requires ModelRunnerOutput-like outputs.")

    a_ids = [str(x) for x in list(getattr(out_a, "req_ids", []) or [])]
    b_ids = [str(x) for x in list(getattr(out_b, "req_ids", []) or [])]
    a_map = {rid: i for i, rid in enumerate(a_ids)}
    b_map = {rid: i for i, rid in enumerate(b_ids)}
    all_ids = set(a_ids) | set(b_ids)
    req_ids = [rid for rid in original_order if rid in all_ids]
    req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}

    def _pick_list_attr(obj: Any, attr: str, idx: int, default: Any) -> Any:
        vals = getattr(obj, attr, None)
        if vals is None:
            return default
        try:
            return vals[idx]
        except Exception:
            return default

    sampled_token_ids: list[list[int]] = []
    pooler_output: list[Optional[Any]] = []
    spec_token_ids_needed = (getattr(out_a, "spec_token_ids", None) is not None) or (
        getattr(out_b, "spec_token_ids", None) is not None
    )
    spec_token_ids: Optional[list[list[int]]] = [] if spec_token_ids_needed else None

    for rid in req_ids:
        if rid in a_map:
            idx = a_map[rid]
            sampled = list(_pick_list_attr(out_a, "sampled_token_ids", idx, []))
            pool_val = _pick_list_attr(out_a, "pooler_output", idx, None)
            spec_val = (
                list(_pick_list_attr(out_a, "spec_token_ids", idx, []))
                if spec_token_ids_needed
                else []
            )
        else:
            idx = b_map[rid]
            sampled = list(_pick_list_attr(out_b, "sampled_token_ids", idx, []))
            pool_val = _pick_list_attr(out_b, "pooler_output", idx, None)
            spec_val = (
                list(_pick_list_attr(out_b, "spec_token_ids", idx, []))
                if spec_token_ids_needed
                else []
            )
        sampled_token_ids.append(sampled)
        pooler_output.append(pool_val)
        if spec_token_ids is not None:
            spec_token_ids.append(spec_val)

    logprobs = None
    if getattr(out_a, "logprobs", None) is not None or getattr(out_b, "logprobs", None) is not None:
        lp_cls = type(getattr(out_a, "logprobs", None) or getattr(out_b, "logprobs", None))
        a_lp = getattr(out_a, "logprobs", None)
        b_lp = getattr(out_b, "logprobs", None)
        logprob_token_ids = []
        logprobs_vals = []
        sampled_token_ranks = []
        for rid in req_ids:
            if rid in a_map and a_lp is not None:
                i = a_map[rid]
                logprob_token_ids.append(list(a_lp.logprob_token_ids[i]))
                logprobs_vals.append(list(a_lp.logprobs[i]))
                sampled_token_ranks.append(int(a_lp.sampled_token_ranks[i]))
            elif rid in b_map and b_lp is not None:
                i = b_map[rid]
                logprob_token_ids.append(list(b_lp.logprob_token_ids[i]))
                logprobs_vals.append(list(b_lp.logprobs[i]))
                sampled_token_ranks.append(int(b_lp.sampled_token_ranks[i]))
            else:
                logprob_token_ids.append([])
                logprobs_vals.append([])
                sampled_token_ranks.append(0)
        logprobs = lp_cls(logprob_token_ids, logprobs_vals, sampled_token_ranks)

    prompt_logprobs_dict = {}
    prompt_logprobs_dict.update(getattr(out_a, "prompt_logprobs_dict", {}) or {})
    prompt_logprobs_dict.update(getattr(out_b, "prompt_logprobs_dict", {}) or {})

    num_nans_in_logits = {}
    num_nans_in_logits.update(getattr(out_a, "num_nans_in_logits", {}) or {})
    num_nans_in_logits.update(getattr(out_b, "num_nans_in_logits", {}) or {})

    out_cls = type(out_a)
    return out_cls(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        spec_token_ids=spec_token_ids,
        logprobs=logprobs,
        prompt_logprobs_dict=prompt_logprobs_dict,
        pooler_output=pooler_output,
        kv_connector_output=merge_kv_connector_outputs(
            getattr(out_a, "kv_connector_output", None),
            getattr(out_b, "kv_connector_output", None),
        ),
        num_nans_in_logits=num_nans_in_logits or None,
    )
