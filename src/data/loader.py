"""Dataset loading helpers."""

from datasets import DatasetDict, concatenate_datasets, get_dataset_config_names, load_dataset, load_from_disk


def _select_split(dataset, split_name):
    if isinstance(dataset, DatasetDict):
        return dataset[split_name]
    return dataset


def _load_hallusionbench(dataset, split_name):
    if not isinstance(dataset, DatasetDict):
        return dataset

    if split_name in dataset:
        ds = dataset[split_name]
        if "source_split" not in ds.column_names:
            ds = ds.add_column("source_split", [split_name] * len(ds))
        return ds

    if split_name in {"all", "test", "full"}:
        merged = []
        for name in ["image", "non_image"]:
            if name in dataset:
                ds = dataset[name]
                if "source_split" not in ds.column_names:
                    ds = ds.add_column("source_split", [name] * len(ds))
                merged.append(ds)
        if not merged:
            raise KeyError(f"HallusionBench split not found: {split_name}")
        return concatenate_datasets(merged)

    raise KeyError(f"HallusionBench split not found: {split_name}")


def _count_mmmu_images(example):
    return sum(1 for idx in range(1, 8) if example.get(f"image_{idx}") is not None)


def _filter_mmmu_single_image(dataset):
    return dataset.filter(lambda example: _count_mmmu_images(example) == 1)


def _load_mmmu_remote(split_name):
    configs = get_dataset_config_names("MMMU/MMMU")
    datasets = [load_dataset("MMMU/MMMU", config, split=split_name) for config in configs]
    return concatenate_datasets(datasets)


def get_dataset(data_name_or_path, tokenizer, prompt_idx, dataset_split="test"):
    """
    Args:
        data_name_or_path: dataset name or path
        tokenizer: tokenizer
        prompt_idx: which query prompt to use
    Returns:
        dataset: dataset
    """

    if "gsm8k" in data_name_or_path:
        try:
            dataset = _select_split(load_from_disk(data_name_or_path), "test")
        except Exception:
            dataset = load_dataset("openai/gsm8k", "socratic")["test"]

    elif "MATH-500" in data_name_or_path:
        try:
            dataset = _select_split(load_from_disk(data_name_or_path), "test")
        except Exception:
            dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]

    elif "AIME_2024" in data_name_or_path:
        try:
            dataset = _select_split(load_from_disk(data_name_or_path), "train")
        except Exception:
            dataset = load_dataset("Maxwell-Jia/AIME_2024")["train"]

    elif "MathVista" in data_name_or_path:
        try:
            dataset = _select_split(load_from_disk(data_name_or_path), dataset_split)
        except Exception:
            dataset = load_dataset("AI4Math/MathVista", split=dataset_split)

    elif "MMStar" in data_name_or_path:
        try:
            dataset = _select_split(load_from_disk(data_name_or_path), dataset_split)
        except Exception:
            dataset = load_dataset("Lin-Chen/MMStar", split=dataset_split)

    elif "LogicVista" in data_name_or_path:
        try:
            dataset = _select_split(load_from_disk(data_name_or_path), dataset_split)
        except Exception:
            dataset = load_dataset("Ruiyang-061X/LogicVista-full", split=dataset_split)

    elif "RealWorldQA" in data_name_or_path:
        try:
            dataset = _select_split(load_from_disk(data_name_or_path), dataset_split)
        except Exception:
            dataset = load_dataset("lmms-lab/RealWorldQA", split=dataset_split)

    elif "ScienceQA-IMG" in data_name_or_path or "ScienceQA" in data_name_or_path:
        try:
            dataset = _select_split(load_from_disk(data_name_or_path), dataset_split)
        except Exception:
            dataset = load_dataset("lmms-lab/ScienceQA-IMG", split=dataset_split)

    elif "HallusionBench" in data_name_or_path:
        try:
            dataset = _load_hallusionbench(load_from_disk(data_name_or_path), dataset_split)
        except Exception:
            dataset = _load_hallusionbench(load_dataset("lmms-lab/HallusionBench"), dataset_split)

    elif "MMMU" in data_name_or_path:
        try:
            dataset = _select_split(load_from_disk(data_name_or_path), dataset_split)
        except Exception:
            dataset = _load_mmmu_remote(dataset_split)
        if "single-image" not in data_name_or_path.lower():
            dataset = _filter_mmmu_single_image(dataset)

    else:
        raise ValueError(f"Unsupported dataset: {data_name_or_path}")

    return dataset
