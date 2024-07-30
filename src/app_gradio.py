"""Text query images based on cn-clip model

- Load the cnclip model (finetuned)
- Encode the images (user dataset)
- Store the clip embeddings of images
- Index the images using FAISS
- Perform the semantic search
- Gradio to interact with user

"""

import cn_clip.clip as clip
import gradio as gr
import numpy as np
import torch
from PIL import Image

from data import get_image_path
from data.faiss_index import (create_faiss_index, load_faiss_index,
                              save_faiss_index)
from data.image_processing import encode_images
from model.utils import load_model


device = "cuda" if torch.cuda.is_available() else "cpu"
img_embs_file_path = "data/processed/fmh_weixin_images_embedding.index"

EMB_MODEL, preprocess = load_model(for_inference=True)
image_paths = get_image_path.read(no_posix=True)


try:
    index = load_faiss_index(img_embs_file_path)
    print(f"Load image index (FAISS) from {img_embs_file_path}")
except (RuntimeError, FileNotFoundError):
    print("No faiss-index of image embeddings found. Create one now.")
    img_embeddings = encode_images(
        image_paths,
        batch_size=32,
        preprocess=preprocess,
        emb_model=EMB_MODEL,
        device=device,
    )
    index = create_faiss_index(np.array(img_embeddings))
    save_faiss_index(index, file_path=img_embs_file_path)


def encode_text(texts):  # the batch version did not work well for gradio
    text = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = EMB_MODEL.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.detach().cpu().numpy()


def search_interface(text_input, topk):
    text_embeddings = encode_text(text_input)
    distances, indices = index.search(text_embeddings, k=topk)
    images = [Image.open(image_paths[idx]) for idx in indices[0]]
    return images

    #  # Assume `results` is a list of image paths or PIL Images
    #  images = [Image.open(img) if isinstance(img, str) else img for img in results]
    #  return images


# Define the Gradio interface
#  interface = gr.Interface(
#      fn=search_interface,
#      inputs=gr.Textbox(lines=2, placeholder="Enter a description...", label="Query"),
#      outputs=gr.Gallery(label="Search Results", columns=2, object_fit="contain"),
#      title="Image Search with Text",
#      description="Enter a description to search for related images using CLIP.",
#  )
interface = gr.Interface(
    fn=search_interface,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter a description...", label="Query"),
        gr.Slider(
            minimum=1, maximum=20, step=1, value=5, label="Number of Results"
        ),  # Slider for topk
    ],
    outputs=gr.Gallery(label="Search Results", columns=2, object_fit="contain"),
    title="Image Search with Text",
    description="Enter a description to search for related images using CLIP.",
)


if __name__ == "__main__":
    interface.launch()
