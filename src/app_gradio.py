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

EMB_MODEL, PREPROCESS = load_model(for_inference=True)
IMAGE_PATHS = get_image_path.read(as_posix=True)


try:
    index = load_faiss_index(img_embs_file_path)
    print(f"Load image index (FAISS) from {img_embs_file_path}")
except (RuntimeError, FileNotFoundError):
    print("No faiss-index of image embeddings found. Create one now.")
    img_embeddings = encode_images(
        IMAGE_PATHS,
        batch_size=32,
        preprocess=PREPROCESS,
        emb_model=EMB_MODEL,
        device=device,
    )
    index = create_faiss_index(np.array(img_embeddings))
    save_faiss_index(index, file_path=img_embs_file_path)


def encode_image(image):
    #  if isinstance(image_path, str):
    #      image = PREPROCESS(Image.open(image_path)).unsqueeze(0).to(device)
    #  elif isinstance(image_path, Image.Image):
    #      image = PREPROCESS(image_path).unsqueeze(0).to(device)
    #  else:
    #      raise ValueError("")

    image = PREPROCESS(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = EMB_MODEL.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.detach().cpu().numpy()


def encode_text(texts):  # the batch version did not work well for gradio
    text = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = EMB_MODEL.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.detach().cpu().numpy()


def search_interface(text_input, image_input, topk):
    try:
        # Check if both inputs are provided
        if text_input and image_input is not None:
            raise ValueError(
                "Please provide either a text query or an image, not both."
            )

        if text_input:
            query_embeddings = encode_text(text_input)
        elif image_input:
            if isinstance(image_input, Image.Image):
                query_embeddings = encode_image(image_input)
            else:
                return "Invalid image data. Please upload a valid image."
        else:
            return "Please provide either a text query or an image."

        distances, indices = index.search(query_embeddings, k=topk)
        images = [Image.open(IMAGE_PATHS[idx]) for idx in indices[0]]
        return images
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Define the Gradio interface
interface = gr.Interface(
    fn=search_interface,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter a description...", label="Query"),
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=1, maximum=20, step=1, value=5, label="Number of Results"),
    ],
    outputs=gr.Gallery(label="Search Results", columns=1, object_fit="contain"),
    title="Image Search with Text or Image",
    description="Enter a description or upload an image to search for related images using CLIP.",
)


if __name__ == "__main__":
    interface.launch()
