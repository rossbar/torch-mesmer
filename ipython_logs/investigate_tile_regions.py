import matplotlib.pyplot as plt
plt.ion()
from torch_mesmer.mesmer import untile_output
from torch_mesmer.mesmer import mesmer_preprocess
from torch_mesmer.mesmer import resize_input
import torch
from torch_mesmer.mesmer import tile_input
import napari
import numpy as np
from pathlib import Path
from torch_mesmer.model import PanopticNet

device = "cuda:1"
datapath = Path("/data2/tissuenet")
dataset = np.load(datapath / "train_512.npz")
X = dataset["X"]
y = dataset["y"]

idx = 5
img, (wcgt, ngt) = X[idx], y[idx].transpose((2, 0, 1))

nim = napari.view_image(img, channel_axis=-1);
nim.add_labels(wcgt);
nim.add_labels(ngt);

dummy = torch.rand(1, 2, 256, 256).to(device)
model = PanopticNet(
            crop_size=256,
            backbone='resnet50',
            pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
            backbone_levels=['C3', 'C4', 'C5'],
            n_semantic_classes=[1, 3, 1, 3],
        ).to(device).eval()
model(dummy)
checkpoint = torch.load("/home/administrator/saved_model_best_dict.pth")
model.load_state_dict(checkpoint)

img = mesmer_preprocess(img[np.newaxis,...])
tiles, tile_meta = tile_input(img, model_image_shape=(256, 256))
tiles = tiles.transpose((0, -1, 1, 2))

tile_batch = torch.tensor(tiles).to(device)
with torch.inference_mode():
    pred = model(tile_batch)

pred = pred.detach().cpu()

# View model outputs for a single tile

t = tiles[0]
p = pred[0]
nim = napari.view_image(t, channel_axis=0, name=("nuc", "wc"))
pred_descr = ("nuc_inner_distance", "nuc_bdry", "nuc_fg", "nuc_bgnd", "wc_id", "wc_bdry", "wc_fg", "wc_bgnd")
for lyr, descr in zip(p, pred_descr):
    nim.add_image(lyr, name=descr, blending="additive");

# Now for a different tile

p = pred[1]
t = tiles[1]
nim = napari.view_image(t, channel_axis=0, name=("nuc", "wc"))
for lyr, descr in zip(p, pred_descr):
    nim.add_image(lyr, name=descr, blending="additive");

# Closer look at overlap regions

p0_rhs = pred[0][..., 192:256]
p0_rhs.shape
p1_lhs = pred[1][..., 0:64]
p1_lhs.shape

# Visualize overlap regions

nim = napari.view_image(p0_rhs, name=pred_descr, channel_axis=0, blending="additive");
nim = napari.view_image(p1_lhs, name=pred_descr, channel_axis=0, blending="additive");

# Compare distribution of pixel values in overlap region

plt.plot(p0_rhs[0].ravel(), p1_lhs[0].ravel(), 'x');

# Look at outputs of untiling (which is where interpolation happens)

untiled = untile_output(pred.numpy().transpose((0, 2, 3, 1)), tile_meta, (256, 256))
nim = napari.view_image(untiled.squeeze(), channel_axis=-1, name=pred_descr);
