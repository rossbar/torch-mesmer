import warnings
import time
from torch_mesmer.mesmer import Mesmer
import numpy as np
from pathlib import Path
import zarr

warnings.simplefilter("ignore")

zname = Path.home() / "hubmap-to-zarr/tissuenet.zarr"
z = zarr.open(zname, mode="r")
k = "HBM222_WQKC_382"
model_path = Path.home() / "saved_model_best_dict_849700.pth"

ds = z[k]
img = ds["image"][:]
chnames = ds["image"].attrs["channels"]
nuc, mem = ds.attrs["nuclear_channel"], ds.attrs["membrane_channel"]
mpp = ds["image"].attrs["mpp"]

im = np.stack(
    [img[chnames.index(nuc)], img[chnames.index(mem)]],
    axis=0,
)
app = Mesmer(model_path=model_path, device="cuda:1")

tic = time.time()
mask = app.predict(im[np.newaxis, ...], image_mpp=mpp, batch_size=4, compartment="whole-cell")[0].squeeze()
toc = time.time()

print(toc - tic)
