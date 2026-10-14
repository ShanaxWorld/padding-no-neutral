"""`self`: own-canvas masking for batch-invariant object detection.

Every image in a padded batch reproduces its own batch-1 computation (paper, Sect. 6).
The code between the CORE markers is copied verbatim from the notebooks that produced
every number in the paper (`notebooks/final_fullval_group*.ipynb`, cell "The `self` fix").
The wrappers below it (`enable_self_torchvision`, `enable_self_hf`) only call
`Fix.begin` / `Fix.end` around each forward pass, as the notebooks' detector classes do.
"""
import math
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------- CORE (verbatim)
class Fix:
    COMPONENTS = ("conv", "norm", "own", "exactmask")
    def __init__(self):
        for c in self.COMPONENTS:
            setattr(self, c, False)
        self.valid, self.canvas, self.trivial, self._cache = None, None, True, {}

    def set(self, components):
        for c in self.COMPONENTS:
            setattr(self, c, c in components)

    def any_on(self):
        return any(getattr(self, c) for c in self.COMPONENTS)

    def begin(self, valid, canvas):
        self.valid = [tuple(int(v) for v in s) for s in valid]
        self.canvas = tuple(int(c) for c in canvas)
        self.trivial = all(v == self.canvas for v in self.valid)

    def end(self):
        self.valid = None

    def active(self):
        return self.valid is not None and not self.trivial

    def applies(self, x):
        return self.active() and x.dim() == 4 and x.shape[0] == len(self.valid)

    @staticmethod
    def _stride(n, N):
        for s in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            if math.ceil(N / s) == n:
                return s
        return N / n

    def extents(self, H, W):
        """each image's valid (rows, cols) at a feature map of size H x W"""
        sh, sw = self._stride(H, self.canvas[0]), self._stride(W, self.canvas[1])
        return [(math.ceil(h / sh), math.ceil(w / sw)) for h, w in self.valid]

    def mask(self, x):
        H, W = x.shape[-2:]
        key = (H, W, tuple(self.valid), self.canvas, x.device, x.dtype)
        if key not in self._cache:
            m = torch.zeros(len(self.valid), 1, H, W, device=x.device, dtype=x.dtype)
            for i, (h, w) in enumerate(self.extents(H, W)):
                m[i, :, :h, :w] = 1
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = m
        return self._cache[key]

def _mask_conv_inputs(root, fix):
    """Zero the padding at the input of every conv / max-pool: each layer then sees exactly the zeros its
    batch-1 run sees past the canvas edge (activations are >= 0 after ReLU, so max-pool is unaffected)."""
    def pre(mod, inp):
        x = inp[0]
        if fix.conv and fix.applies(x):
            return (x * fix.mask(x),) + tuple(inp[1:])
    for m in root.modules():
        if isinstance(m, (nn.Conv2d, nn.MaxPool2d)):
            m.register_forward_pre_hook(pre)

def _masked_groupnorm(root, fix):
    """GroupNorm statistics over each image's own valid area only."""
    n = 0
    for mod in root.modules():
        if isinstance(mod, nn.GroupNorm):
            orig = mod.forward
            def fwd(x, mod=mod, orig=orig):
                if not (fix.norm and fix.applies(x)):
                    return orig(x)
                m = fix.mask(x).float(); B, C, H, W = x.shape; G = mod.num_groups
                xg, mg = x.float().reshape(B, G, C // G, H, W), m.reshape(B, 1, 1, H, W)
                cnt = mg.sum(dim=(2, 3, 4), keepdim=True) * (C // G)
                mean = (xg * mg).sum(dim=(2, 3, 4), keepdim=True) / cnt
                var = (((xg - mean) ** 2) * mg).sum(dim=(2, 3, 4), keepdim=True) / cnt
                y = ((xg - mean) / torch.sqrt(var + mod.eps)).reshape(B, C, H, W)
                if mod.affine:
                    y = y * mod.weight.float()[None, :, None, None] + mod.bias.float()[None, :, None, None]
                return y.to(x.dtype)
            mod.forward = fwd; n += 1
    return n

def _suppress_padding_scores(conv, fix):
    """Classification logits at positions that do not exist in the image's batch-1 run -> -1e4."""
    def hook(mod, inp, out):
        if fix.conv and fix.applies(out):
            return out.masked_fill(fix.mask(out) == 0, -1e4)
    conv.register_forward_hook(hook)

def _own_anchors(ag, fix):
    """Each image gets the anchors its own batch-1 canvas would give: torchvision spaces anchors at
    canvas // grid, so we use the image's own canvas and grid. Anchors at positions that do not exist in
    its batch-1 run become zero-size boxes, which torchvision removes."""
    orig = ag.forward
    def fwd(image_list, feature_maps):
        if not (fix.own and fix.active()):
            return orig(image_list, feature_maps)
        grid_sizes = [list(f.shape[-2:]) for f in feature_maps]
        Hc, Wc = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        ag.set_cell_anchors(dtype, device)
        n_cell = [ca.shape[0] for ca in ag.cell_anchors]
        out = []
        for ch, cw in fix.valid:
            strides, keep = [], []
            for (gh, gw), a in zip(grid_sizes, n_cell):
                sh, sw = 2 ** round(math.log2(Hc / gh)), 2 ** round(math.log2(Wc / gw))
                oh, ow = math.ceil(ch / sh), math.ceil(cw / sw)
                strides.append([torch.tensor(ch // oh, dtype=torch.int64, device=device),
                                torch.tensor(cw // ow, dtype=torch.int64, device=device)])
                k = torch.zeros(gh, gw, dtype=dtype, device=device); k[:oh, :ow] = 1
                keep.append(k.reshape(-1).repeat_interleave(a))
            anc = ag.grid_anchors(grid_sizes, strides)
            out.append(torch.cat([x * k[:, None] for x, k in zip(anc, keep)]))
        return out
    ag.forward = fwd

def install_fix_tv(model, fix):
    heads = [model.rpn.head] if hasattr(model, "rpn") else [model.head]
    for root in [model.backbone] + heads:
        _mask_conv_inputs(root, fix)
    has_roi = hasattr(model, "roi_heads")
    def zero_outputs(mod, inp, out):
        if not (fix.conv and fix.active()):
            return
        new = type(out)()
        for k, v in out.items():
            v = v * fix.mask(v)
            if fix.own and has_roi:
                # RoIAlign at batch 1 clamps samples at the canvas edge to the last row/column; copying that
                # row/column one step outward reproduces this inside a bigger canvas.
                H, W = v.shape[-2:]
                for i, (hl, wl) in enumerate(fix.extents(H, W)):
                    if wl < W:
                        v[i, :, :hl, wl] = v[i, :, :hl, wl - 1]
                    if hl < H:
                        v[i, :, hl, :min(wl + 1, W)] = v[i, :, hl - 1, :min(wl + 1, W)]
            new[k] = v
        return new
    model.backbone.register_forward_hook(zero_outputs)
    g = sum(_masked_groupnorm(h, fix) for h in heads) + _masked_groupnorm(model.backbone, fix)
    if hasattr(model, "rpn"):
        _own_anchors(model.rpn.anchor_generator, fix); _suppress_padding_scores(model.rpn.head.cls_logits, fix)
    else:
        _own_anchors(model.anchor_generator, fix); _suppress_padding_scores(model.head.classification_head.cls_logits, fix)
    fix.SELF = ("conv", "norm", "own") if g else ("conv", "own")

def install_fix_hf(model, fix):
    enc = model.model.backbone
    _mask_conv_inputs(enc, fix)
    orig = enc.forward
    def fwd(pixel_values, pixel_mask):
        out = orig(pixel_values, pixel_mask)
        if not (fix.exactmask and fix.active()):
            return out
        # the transformer's padding mask with exact extents (the library shrinks it with nearest-neighbour rounding)
        return [(fm, fix.mask(fm)[:, 0].bool()) for fm, _ in out]
    enc.forward = fwd
    fix.SELF = ("conv", "exactmask")

# ----------------------------------------------------------------------------- end of CORE


def _roundup(x, d):
    return int(math.ceil(x / d) * d)


def enable_self_torchvision(model):
    """Install `self` on a torchvision detector (Faster/Mask/Keypoint R-CNN, RetinaNet, FCOS).

    The model keeps its normal call signature, model(list_of_image_tensors). Inside a padded batch,
    every image then gets its batch-1 output (up to floating-point summation order).
    Returns the `Fix` object; `fix.set(())` turns the fix off, `fix.set(fix.SELF)` back on.
    """
    fix = Fix()
    install_fix_tv(model, fix)
    fix.set(fix.SELF)
    transform = model.transform
    orig_transform_forward = transform.forward

    def transform_forward(images, targets=None):
        image_list, targets = orig_transform_forward(images, targets)
        if fix.any_on():
            d = transform.size_divisible
            sizes = [tuple(int(v) for v in s) for s in image_list.image_sizes]
            valid = [(_roundup(h, d), _roundup(w, d)) for h, w in sizes] if fix.own else sizes
            fix.begin(valid, tuple(image_list.tensors.shape[-2:]))
        return image_list, targets

    transform.forward = transform_forward
    orig_forward = model.forward

    def forward(*args, **kwargs):
        try:
            return orig_forward(*args, **kwargs)
        finally:
            fix.end()

    model.forward = forward
    return fix


def enable_self_hf(model):
    """Install `self` on a Hugging Face DETR-family detector that takes a `pixel_mask`
    (tested: DETR, Conditional DETR). Call the model as usual with pixel_values and pixel_mask."""
    fix = Fix()
    install_fix_hf(model, fix)
    fix.set(fix.SELF)
    orig_forward = model.forward

    def forward(pixel_values=None, pixel_mask=None, **kwargs):
        if pixel_mask is not None and fix.any_on():
            valid = [(int(pixel_mask[i, :, 0].sum()), int(pixel_mask[i, 0, :].sum())) for i in range(pixel_mask.shape[0])]
            fix.begin(valid, tuple(pixel_values.shape[-2:]))
        try:
            return orig_forward(pixel_values=pixel_values, pixel_mask=pixel_mask, **kwargs)
        finally:
            fix.end()

    model.forward = forward
    return fix
