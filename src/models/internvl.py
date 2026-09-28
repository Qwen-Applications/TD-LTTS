"""Lightweight InternVL chat adapter for the ECL-TTS evaluation code.

The local InternVL3.5 checkpoint ships remote model code but no usable
Transformers AutoProcessor in this environment.  This adapter implements the
small processor surface used by the benchmark code: chat-template formatting,
PIL image preprocessing, and multimodal token insertion.
"""

import math
from types import MethodType

import numpy as np
import torch
from PIL import Image


IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMAGE_PLACEHOLDER = "<image>"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def is_internvl_config(config):
    return getattr(config, "model_type", None) == "internvl_chat"


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height), Image.BICUBIC)

    processed_images = []
    for block_idx in range(blocks):
        x0 = (block_idx % (target_width // image_size)) * image_size
        y0 = (block_idx // (target_width // image_size)) * image_size
        box = (x0, y0, x0 + image_size, y0 + image_size)
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size), Image.BICUBIC))
    return processed_images


def _image_to_tensor(image, image_size):
    if not isinstance(image, Image.Image):
        raise TypeError(f"InternVLProcessor expects PIL.Image inputs, got {type(image)!r}")
    image = image.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class InternVLProcessor:
    """Subset of the Qwen-style processor API required by this repo."""

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.image_size = int(config.force_image_size or config.vision_config.image_size)
        patch_size = int(config.vision_config.patch_size)
        self.num_image_token = int((self.image_size // patch_size) ** 2 * (config.downsample_ratio**2))
        self.min_dynamic_patch = int(getattr(config, "min_dynamic_patch", 1))
        self.max_dynamic_patch = int(getattr(config, "max_dynamic_patch", 12))
        self.use_thumbnail = bool(getattr(config, "use_thumbnail", True))

    @classmethod
    def from_pretrained(cls, model_name_or_path, tokenizer, config):
        return cls(tokenizer=tokenizer, config=config)

    def _content_to_text(self, content):
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content or "")

        parts = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type == "image":
                parts.append(IMAGE_PLACEHOLDER)
            elif item_type == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **tokenizer_kwargs):
        system_message = None
        body = []
        for message in messages:
            role = message.get("role", "user")
            content = self._content_to_text(message.get("content", ""))
            if role == "system" and system_message is None:
                system_message = content
            else:
                body.append((role, content))

        chunks = []
        if system_message:
            chunks.append(f"<|im_start|>system\n{system_message}<|im_end|>\n")
        for role, content in body:
            chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            chunks.append("<|im_start|>assistant\n")
        prompt = "".join(chunks)

        if tokenize:
            return self.tokenizer(prompt, **tokenizer_kwargs)
        return prompt

    def _preprocess_images(self, images):
        all_tensors = []
        num_patches = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise TypeError(f"InternVLProcessor expects PIL.Image inputs, got {type(image)!r}")
            crops = _dynamic_preprocess(
                image.convert("RGB"),
                min_num=self.min_dynamic_patch,
                max_num=self.max_dynamic_patch,
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail,
            )
            tensors = [_image_to_tensor(crop, self.image_size) for crop in crops]
            stacked = torch.stack(tensors)
            all_tensors.append(stacked)
            num_patches.append(stacked.shape[0])

        return torch.cat(all_tensors, dim=0), num_patches

    def _replace_image_placeholders(self, prompt, num_patches):
        if IMAGE_PLACEHOLDER not in prompt:
            prompt = f"{IMAGE_PLACEHOLDER}\n{prompt}"

        for patches in num_patches:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * (self.num_image_token * patches) + IMG_END_TOKEN
            prompt = prompt.replace(IMAGE_PLACEHOLDER, image_tokens, 1)
        return prompt.replace(IMAGE_PLACEHOLDER, "")

    def __call__(self, images=None, text=None, return_tensors="pt", padding=True, **kwargs):
        if text is None:
            text = [""]
        elif isinstance(text, str):
            text = [text]
        else:
            text = list(text)

        if images is None:
            return self.tokenizer(text, return_tensors=return_tensors, padding=padding, **kwargs)

        if isinstance(images, Image.Image):
            images = [images]
        else:
            images = list(images)
        if len(text) != 1:
            raise NotImplementedError("InternVLProcessor currently supports one text prompt per image batch.")

        pixel_values, num_patches = self._preprocess_images(images)
        prompt = self._replace_image_placeholders(text[0], num_patches)
        encoded = self.tokenizer([prompt], return_tensors=return_tensors, padding=padding, **kwargs)
        encoded["pixel_values"] = pixel_values.to(torch.bfloat16)
        encoded["image_flags"] = torch.ones(pixel_values.shape[0], 1, dtype=torch.long)
        return encoded


def _sequence_has_prefix(sequence, prefix):
    if sequence.shape[0] < prefix.shape[0]:
        return False
    return torch.equal(sequence[: prefix.shape[0]], prefix.to(sequence.device))


def _prefix_sequences_if_needed(sequences, input_ids):
    if sequences is None or input_ids is None:
        return sequences
    if sequences.ndim != 2 or input_ids.ndim != 2:
        return sequences
    if sequences.shape[0] != input_ids.shape[0]:
        return sequences

    rows = []
    changed = False
    for seq, prefix in zip(sequences, input_ids):
        if _sequence_has_prefix(seq, prefix):
            rows.append(seq)
        else:
            rows.append(torch.cat([prefix.to(seq.device), seq], dim=0))
            changed = True
    if not changed:
        return sequences

    max_len = max(row.shape[0] for row in rows)
    pad_id = 0
    padded = sequences.new_full((len(rows), max_len), pad_id)
    for idx, row in enumerate(rows):
        padded[idx, : row.shape[0]] = row
    return padded


def patch_internvl_generation(model):
    """Make InternVL generate outputs look like Qwen generate outputs.

    InternVL's remote `generate` calls the language model with `inputs_embeds`.
    Some Transformers versions then return only newly generated ids, while this
    repo expects `prompt_ids + generated_ids`.  Prefixing keeps all existing
    decoding and latent-prefix code paths unchanged.
    """

    if getattr(model, "_ecl_tts_internvl_generate_patched", False):
        return model

    original_generate = model.generate

    def generate_with_prefixed_sequences(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        kwargs.pop("image_flags", None)
        outputs = original_generate(*args, **kwargs)
        if input_ids is None:
            return outputs

        if hasattr(outputs, "sequences"):
            outputs.sequences = _prefix_sequences_if_needed(outputs.sequences, input_ids)
            return outputs
        if torch.is_tensor(outputs):
            return _prefix_sequences_if_needed(outputs, input_ids)
        return outputs

    model.generate = MethodType(generate_with_prefixed_sequences, model)
    model._ecl_tts_internvl_generate_patched = True
    return model
