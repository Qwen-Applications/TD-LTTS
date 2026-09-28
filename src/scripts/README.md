# Experiment Scripts

## Launchers

- `run_benchmark.sh`: run one benchmark or one explicit shard
- `launch_sharded.sh`: split a benchmark over an arbitrary GPU list
- `launch_token_disentangle.sh`: run the token-disentangled method
- `launch_ablation.sh`: run routing and branch ablations
- `launch_qwen25vl_ablation_sequence.sh`: run Qwen2.5-VL ablations sequentially

Shared model, dataset, and method defaults are defined in
`lib/experiment_config.sh`. Set `LATENT_TTS_MODEL_ROOT` and
`LATENT_TTS_DATA_ROOT`, or pass `--model` and `--data` explicitly.

## Utilities

- `summarize_sharded_results.py`: validate and aggregate completed shards
- `summarize_ablation_suite.py`: summarize an ablation suite
- `prepare_mmmu_single_image_val.py`: prepare the single-image MMMU subset
- `score_hallusionbench_results.py`: compute HallusionBench metrics

Use `--dry-run` on launchers to inspect generated commands.
