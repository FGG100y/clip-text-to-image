"""
Embed the text using CN-CLIP model
"""

from tqdm import tqdm
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import cn_clip.clip as clip


def encode_texts(text, batch_size, emb_model, device):
    dataset = Dataset.from_dict({"raw_text": text})
    dataset = dataset.map(
        lambda batch: {"text": clip.tokenize(batch["raw_text"])},
        remove_columns=["raw_text"],
        batched=True,
    )
    dataset.set_format("torch")
    dataloader = DataLoader(dataset, batch_size=batch_size)

    text_embeddings = []
    pbar = tqdm(total=len(text) // batch_size, position=0)
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            text_embeddings.extend(
                emb_model.encode_text(**batch).detach().cpu().numpy()
            )
            pbar.update(1)
        pbar.close()
    embeddings = np.stack(text_embeddings)
    return embeddings / np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
