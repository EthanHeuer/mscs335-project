import pathlib
import torch
import zipfile


data_dir = pathlib.Path("../../data/maestro-v2.0.0")
if not data_dir.exists():
    torch.hub.download_url_to_file(
        "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip",
        "../../data/maestro-v2.0.0-midi.zip",
    )

    with zipfile.ZipFile("../../data/maestro-v2.0.0-midi.zip", "r") as zip_ref:
        zip_ref.extractall("../../data")
        pathlib.Path("../../data/maestro-v2.0.0-midi.zip").unlink()
