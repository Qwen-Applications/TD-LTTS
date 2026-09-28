"""
Utilities for training-free crop and masked-view construction.
"""

try:
    from PIL import ImageDraw
except Exception:  # pragma: no cover
    ImageDraw = None


def _normalize_grid_sizes(grid_sizes):
    if grid_sizes is None:
        return (2, 3)
    if isinstance(grid_sizes, str):
        parts = [item.strip() for item in grid_sizes.split(",") if item.strip()]
        return tuple(int(item) for item in parts if int(item) > 0)
    return tuple(int(item) for item in grid_sizes if int(item) > 0)


def get_primary_image(media):
    if not media:
        return None
    image = media.get("image")
    if image is not None:
        return image
    images = media.get("images") or []
    if images:
        return images[0]
    return None


def build_view_media(image):
    if image is None:
        return None
    return {
        "image": image,
        "images": [image],
    }


def _dedupe_bbox_key(bbox):
    return tuple(int(value) for value in bbox)


def build_candidate_crops(image, grid_sizes=(2, 3), include_center=True, min_side=32):
    if image is None or not hasattr(image, "crop") or not hasattr(image, "size"):
        return []

    grid_sizes = _normalize_grid_sizes(grid_sizes)
    width, height = image.size
    views = []
    seen = set()

    def add_view(bbox, view_id, kind="crop"):
        left, top, right, bottom = [int(value) for value in bbox]
        if right - left < min_side or bottom - top < min_side:
            return
        key = _dedupe_bbox_key((left, top, right, bottom))
        if key in seen:
            return
        seen.add(key)
        crop = image.crop((left, top, right, bottom))
        views.append(
            {
                "view_id": view_id,
                "kind": kind,
                "bbox": key,
                "image": crop,
                "media": build_view_media(crop),
            }
        )

    for grid_size in grid_sizes:
        if grid_size <= 0:
            continue
        for row in range(grid_size):
            for col in range(grid_size):
                left = int(round(col * width / grid_size))
                right = int(round((col + 1) * width / grid_size))
                top = int(round(row * height / grid_size))
                bottom = int(round((row + 1) * height / grid_size))
                add_view((left, top, right, bottom), f"grid{grid_size}_{row}_{col}")

    if include_center:
        margin_w = int(round(width * 0.15))
        margin_h = int(round(height * 0.15))
        add_view((margin_w, margin_h, width - margin_w, height - margin_h), "center")

    return views


def _resolve_mask_fill(image, fill):
    if isinstance(fill, (tuple, list)):
        base_value = int(fill[0]) if fill else 127
    else:
        base_value = int(fill)

    if image is None:
        return base_value

    mode = getattr(image, "mode", None)
    try:
        bands = image.getbands()
    except Exception:
        bands = None

    num_bands = len(bands) if bands else 1

    if mode == "1":
        return 1 if base_value > 0 else 0
    if num_bands <= 1 or mode in {"L", "I", "F", "P"}:
        return base_value
    if mode == "LA":
        return (base_value, 255)
    if mode == "RGBA":
        return (base_value, base_value, base_value, 255)
    return tuple(base_value for _ in range(num_bands))


def build_masked_view(image, bbox, fill=(127, 127, 127)):
    if image is None or not hasattr(image, "copy") or ImageDraw is None:
        return None
    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    draw.rectangle(bbox, fill=_resolve_mask_fill(masked, fill))
    return masked


def summarize_views(views):
    summary = []
    for view in views or []:
        summary.append(
            {
                "view_id": view.get("view_id"),
                "kind": view.get("kind"),
                "bbox": list(view.get("bbox") or []),
                "support_score": None if view.get("support_score") is None else float(view["support_score"]),
            }
        )
    return summary
