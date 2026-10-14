"""Exact-shape bucketing (paper, Sect. 6.3): batch only images whose batch-1 canvases are identical."""
import math


def exact_shape_buckets(canvas_sizes, batch_size=8):
    """canvas_sizes: list of each image's batch-1 canvas (H, W), i.e. its model-input size rounded up
    to the library's size divisor (32 for torchvision, 1 for Hugging Face DETRs).
    Returns a list of batches (lists of indices) in which no image is padded beyond its own canvas."""
    groups = {}
    for i, s in enumerate(canvas_sizes):
        groups.setdefault(tuple(int(v) for v in s), []).append(i)
    return [g[k:k + batch_size] for g in groups.values() for k in range(0, len(g), batch_size)]


def torchvision_canvas(model, image_tensor):
    """Batch-1 canvas of one image for a torchvision detector (resize, then round up to size_divisible)."""
    import torch
    with torch.inference_mode():
        il, _ = model.transform([image_tensor])
    h, w = il.image_sizes[0]
    d = model.transform.size_divisible
    return (int(math.ceil(h / d) * d), int(math.ceil(w / d) * d))
