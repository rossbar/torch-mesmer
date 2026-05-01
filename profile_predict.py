import warnings
import time
import torch
from torch_mesmer.model_utils import create_model
from torch_mesmer.mesmer import Mesmer
import numpy as np
from pathlib import Path
import zarr

warnings.simplefilter("ignore")

zname = Path.home() / "hubmap-to-zarr/tissuenet.zarr"
z = zarr.open(zname, mode="r")
k = "HBM222_WQKC_382"
device = "cuda:1"
model_path = Path.home() / ".deepcell/models/saved_model_full_8_best_dict.pth"
model, _, _ = create_model()
model.load_state_dict(
    torch.load(model_path, map_location=device, weights_only=True)
)
model.to(device)

ds = z[k]
img = ds["image"][:]
chnames = ds["image"].attrs["channels"]
nuc, mem = ds.attrs["nuclear_channel"], ds.attrs["membrane_channel"]
mpp = ds["image"].attrs["mpp"]

im = np.stack(
    [img[chnames.index(nuc)], img[chnames.index(mem)]],
    axis=-1,
)
app = Mesmer(model=model, device=device)

tic = time.time()
mask = app.predict(im[np.newaxis, ...], image_mpp=mpp).squeeze()
toc = time.time()

print(toc - tic)
