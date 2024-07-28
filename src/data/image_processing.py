"""
Embed the image using CN-CLIP model
"""

from PIL import Image as pImage
from tqdm import tqdm
import numpy as np
import torch
from datasets import Dataset, Image
from torch.utils.data import DataLoader


def encode_images(images, batch_size, preprocess, emb_model, device):

    def transform_fn(examples):
        image_paths = [x["path"] for x in examples["image"]]
        examples["image"] = [
            preprocess(pImage.open(image_path).convert("RGB"))
            #  preprocess(pImage.open(image_path)).unsqueeze(0).to(device)
            for image_path in image_paths
        ]
        return examples

    dataset = Dataset.from_dict({"image": images}).cast_column(
        "image", Image(decode=False)
    )
    dataset.set_format("torch")
    dataset.set_transform(transform_fn)  # preprocessing
    dataloader = DataLoader(dataset, batch_size=batch_size)
    #  breakpoint()

    image_embeddings = []
    pbar = tqdm(total=len(images) // batch_size, position=0)
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            image_embeddings.extend(
                emb_model.encode_image(**batch).detach().cpu().numpy()
            )
            pbar.update(1)
        pbar.close()
    embeddings = np.stack(image_embeddings)
    return embeddings / np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
