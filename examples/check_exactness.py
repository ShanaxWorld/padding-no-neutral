"""Reproduce the core finding in a few minutes on any GPU (or CPU).

    pip install torch torchvision numpy pillow
    python examples/check_exactness.py --images /path/to/coco/val2017 --n 64

Runs a torchvision Faster R-CNN (COCO weights) on N images:
  1. alone (batch size 1)                      -> reference
  2. in random padded batches of 8             -> share of images whose confident detections change
  3. the same batches with `self` enabled      -> should be ~0%
  4. exact-shape buckets                       -> should be ~0%
"""
import argparse, glob, os, random, sys
import numpy as np, torch, torchvision
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from padfix import enable_self_torchvision, change_rate, exact_shape_buckets, torchvision_canvas

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True, help="folder with .jpg images (e.g. COCO val2017)")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--batch", type=int, default=8)
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = "cuda" if torch.cuda.is_available() else "cpu"
W = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=W).to(dev).eval()
tf = W.transforms()
fix = enable_self_torchvision(model)

paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")))[: args.n]
imgs = [tf(Image.open(p).convert("RGB")).to(dev) for p in paths]

def pack(o):
    return {k: o[k].float().cpu().numpy() for k in ("boxes", "scores", "labels")}

@torch.inference_mode()
def run(batches):
    out = [None] * len(imgs)
    for b in batches:
        for i, o in zip(b, model([imgs[i] for i in b])):
            out[i] = pack(o)
    return out

fix.set(())                                   # fix off
ref = run([[i] for i in range(len(imgs))])
order = list(range(len(imgs))); random.Random(0).shuffle(order)
rand = [order[k:k + args.batch] for k in range(0, len(order), args.batch)]
print(f"random batches of {args.batch}, default : {100 * change_rate(ref, run(rand)):.1f}% of images change")
fix.set(fix.SELF)                             # fix on
print(f"random batches of {args.batch}, + self  : {100 * change_rate(ref, run(rand)):.1f}% of images change")
fix.set(())
canv = [torchvision_canvas(model, x) for x in imgs]
print(f"exact-shape buckets           : {100 * change_rate(ref, run(exact_shape_buckets(canv, args.batch))):.1f}% of images change")
