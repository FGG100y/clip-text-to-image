"""Text query images based on cn-clip model

- Load the cnclip model (finetuned)
- Encode the images (user dataset)
- Store the clip embeddings of images
- Index the images using FAISS
- Perform the semantic search

TODO:
    Gradio to interact with user

"""

#  import gradio as gr
import numpy as np
import torch

from data import get_image_path
from data.faiss_index import (create_faiss_index, load_faiss_index,
                              save_faiss_index)
from data.image_processing import encode_images
from data.text_processing import encode_texts
from model.utils import load_model
from visualization.utils import viz

device = "cuda" if torch.cuda.is_available() else "cpu"
img_embs_file_path = "data/processed/fmh_weixin_images_embedding.index"

emb_model, preprocess = load_model(for_inference=True)
image_paths = get_image_path.read(no_posix=True)

text_query = ["四人帮", "女子头像"]
text_embeddings = encode_texts(
    text_query, batch_size=32, emb_model=emb_model, device=device
)

try:
    index = load_faiss_index(img_embs_file_path)
    print(f"Load image index (FAISS) from {img_embs_file_path}")
except (RuntimeError, FileNotFoundError):
    print("No faiss-index of image embeddings found. Create one now.")
    img_embeddings = encode_images(
        image_paths,
        batch_size=32,
        preprocess=preprocess,
        emb_model=emb_model,
        device=device,
    )
    index = create_faiss_index(np.array(img_embeddings))
    save_faiss_index(index, file_path=img_embs_file_path)


def sematic_search(topk=3, vizualization=True):
    distances, indices = index.search(text_embeddings, k=topk)
    text_query_as_title = [(txt,) * topk for txt in text_query]
    for Is, Ds, tt in zip(indices, distances, text_query_as_title):
        indices_distances = list(zip(Is, Ds))
        indices_distances.sort(key=lambda x: x[1])  # Sort based on the distances

        if vizualization:
            viz(indices_distances, image_paths, tt)


if __name__ == "__main__":
    sematic_search()

#  # TODO 找时间完成前端页面（必须找gpt阁下）
#  def search_images(query):
#      if isinstance(query, str):
#          # Handle text query
#          query_embedding = encode_texts(
#              text_query, batch_size=32, preprocess=preprocess, emb_model=emb_model, device=device
#          )
#          distances, indices = index.search(query_embedding.reshape(1,-1), k=5)
#      else:
#          # Handle image query
#          img_embedding = encode_images(
#              image_path, batch_size=32, preprocess=preprocess, emb_model=emb_model, device=device
#          )
#          distances, indices = index.search(img_embedding, k=5)
#
#      # Get image paths and descriptions
#      results = [(image_paths[idx], descriptions[idx]) for idx in indices[0]]
#      return results
#
#
#  def search_interface(text_query, image_query):
#      if text_query:
#          # Text search
#          results = search_images(text_query)
#      elif image_query:
#          # Image search
#          results = search_images(image_query)
#          #  # Provide a button to save the image
#          #  add_image_button = gr.Button("Save Image for Later")
#          #  add_image_button.click(
#          #      fn=add_image_to_index, inputs=image_query, outputs="text"
#          #  )
#      else:
#          results = []
#
#      return results
#
#
#  interface = gr.Interface(
#      fn=search_interface,
#      inputs=[
#          gr.Textbox(lines=1, placeholder="Search..."),
#          gr.Image(type="pil"),
#      ],
#      outputs=gr.Gallery(label="Search Results"),
#  )
#
#
#  if __name__ == "__main__":
#      interface.launch()
