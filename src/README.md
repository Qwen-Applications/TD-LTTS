# Source Layout

- `main.py`: model loading, inference, evaluation, and result persistence
- `benchmarks/`: benchmark-specific prompts, parsing, and acceptance helpers
- `data/`: dataset loading and multimodal example preparation
- `methods/latentgrad/`: latent generation and optimization
- `methods/baselines/`: output-space and latent-refinement baselines
- `rewards/`: reasoning and visual reward implementations
- `evaluation/`: answer extraction, finalization, and judging
- `models/`: model-family adapters
- `prompts/`: reusable prompt templates
- `scripts/`: parameterized experiment launchers and result aggregation

New benchmark logic should stay under `benchmarks/`; optimization and routing
changes belong under `methods/`; reward definitions belong under `rewards/`.
Keep `main.py` focused on orchestration.
