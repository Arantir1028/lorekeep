from __future__ import annotations

from typing import Any, Callable

from engine.hijack.types import _PatchState

RequestIdGetter = Callable[[Any], str | None]


def build_public_seq_group_metadata(
    *,
    scheduler_obj: Any,
    scheduled_seq_group: Any,
    scheduler_outputs: Any,
    now: float,
    state: _PatchState,
    sequence_status_cls: Any,
    sequence_group_metadata_cls: Any,
    sequence_group_metadata_delta_cls: Any,
    request_id_getter: RequestIdGetter,
) -> Any:
    seq_group = scheduled_seq_group.seq_group
    token_chunk_size = scheduled_seq_group.token_chunk_size
    seq_group.maybe_set_first_scheduled_time(now)
    trace_request_id = request_id_getter(seq_group)
    trace_is_prompt = None
    trace_num_computed_tokens = None
    try:
        trace_is_prompt = bool(seq_group.is_prefill())
    except Exception:
        trace_is_prompt = None
    try:
        seqs_for_trace = seq_group.get_seqs()
        if seqs_for_trace:
            trace_num_computed_tokens = int(
                seqs_for_trace[0].data.get_num_computed_tokens()
            )
    except Exception:
        trace_num_computed_tokens = None
    if trace_request_id:
        state.metrics.record_phase1_step_trace(
            request_id=str(trace_request_id),
            event="public_schedule_group",
            is_prefill=trace_is_prompt,
            token_chunk_size=int(token_chunk_size),
            num_computed_tokens=trace_num_computed_tokens,
        )

    seq_group_metadata = scheduler_obj._seq_group_metadata_cache[scheduler_obj.cache_id].get_object()
    seq_group_metadata.seq_data.clear()
    seq_group_metadata.block_tables.clear()

    seq_data: dict[int, Any] = {}
    block_tables: dict[int, list[int]] = {}

    if seq_group.is_encoder_decoder():
        encoder_seq = seq_group.get_encoder_seq()
        assert encoder_seq is not None
        encoder_seq_data = encoder_seq.data
        cross_block_table = scheduler_obj.block_manager.get_cross_block_table(seq_group)
    else:
        encoder_seq_data = None
        cross_block_table = None

    for seq in seq_group.get_seqs(status=sequence_status_cls.RUNNING):
        seq_id = seq.seq_id
        seq_data[seq_id] = seq.data
        block_tables[seq_id] = scheduler_obj.block_manager.get_block_table(seq)
        scheduler_obj.block_manager.access_all_blocks_in_seq(seq, now)

    if scheduler_obj.cache_config.enable_prefix_caching:
        common_computed_block_nums = scheduler_obj.block_manager.get_common_computed_block_ids(
            seq_group.get_seqs(status=sequence_status_cls.RUNNING)
        )
    else:
        common_computed_block_nums = []

    do_sample = True
    is_prompt = seq_group.is_prefill()
    is_first_prefill = False
    if is_prompt:
        seqs = seq_group.get_seqs()
        assert len(seqs) == 1
        num_computed_tokens = seqs[0].data.get_num_computed_tokens()
        is_first_prefill = num_computed_tokens == 0
        if token_chunk_size + num_computed_tokens < seqs[0].data.get_len():
            do_sample = False

    if is_first_prefill or not scheduler_obj.scheduler_config.send_delta_data:
        return sequence_group_metadata_cls(
            request_id=seq_group.request_id,
            is_prompt=is_prompt,
            seq_data=seq_data,
            sampling_params=seq_group.sampling_params,
            block_tables=block_tables,
            do_sample=do_sample,
            pooling_params=seq_group.pooling_params,
            token_chunk_size=token_chunk_size,
            lora_request=seq_group.lora_request,
            computed_block_nums=common_computed_block_nums,
            encoder_seq_data=encoder_seq_data,
            cross_block_table=cross_block_table,
            state=seq_group.state,
            token_type_ids=seq_group.token_type_ids,
            multi_modal_data=(
                seq_group.multi_modal_data
                if scheduler_outputs.num_prefill_groups > 0 else None
            ),
            multi_modal_placeholders=(
                seq_group.multi_modal_placeholders
                if scheduler_outputs.num_prefill_groups > 0 else None
            ),
        )

    seq_data_delta = {}
    for seq_id, data in seq_data.items():
        seq_data_delta[seq_id] = data.get_delta_and_reset()
    return sequence_group_metadata_delta_cls(
        seq_data_delta,
        seq_group.request_id,
        block_tables,
        is_prompt,
        do_sample=do_sample,
        token_chunk_size=token_chunk_size,
        computed_block_nums=common_computed_block_nums,
    )
