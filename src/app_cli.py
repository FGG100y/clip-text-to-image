"""Text query images based on cn-clip model

- Load the cnclip model (finetuned)
- Encode the images (user dataset)
- Store the clip embeddings of images
- Index the images using FAISS
- Perform the semantic search

"""

import numpy as np
import torch

from data import get_image_path
from data.faiss_index import (create_faiss_index, load_faiss_index,
                              save_faiss_index)
from data.image_processing import encode_images
from data.text_processing import encode_texts
from model.audio import stt_model
from model.utils import load_model
from visualization.utils import viz_subplot

device = "cuda" if torch.cuda.is_available() else "cpu"
img_embs_file_path = "data/processed/fmh_weixin_images_embedding.index"

EMB_MODEL, preprocess = load_model(for_inference=True)
image_paths = get_image_path.read(as_posix=True)


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


def main(using_stt=True, text_query=("四人帮", "小孩子在玩耍"), topk=1):
    if using_stt:
        exit_words_l = [
            # 都是朋友时
            "再见",
            "再見",
            "先这样",
            "byebye",
            # 惹人生气时
            "滚蛋",
            "跪安吧",
            "退下吧",
        ]
        faster_whisper = stt_model.load_faster_whisper()
        while True:
            try:
                speech2text = stt_model.transcribe_fast(
                    model=faster_whisper, duration=35, adjust_duration=3, verbose=1
                )
                if not isinstance(speech2text, tuple):
                    text_query = tuple([speech2text])
            except Exception as e:
                print(e)
            else:
                say_goodbye = [w for w in exit_words_l if w in speech2text]
                if len(say_goodbye) > 0:
                    print(say_goodbye)
                    break
            # searching
            sematic_search(text_query, topk=topk, vizualization=True)
    else:
        # searching
        sematic_search(text_query, topk=topk, vizualization=True)


def sematic_search(text_query, topk=3, vizualization=True):
    text_embeddings = encode_texts(
        text_query, batch_size=32, emb_model=EMB_MODEL, device=device
    )
    distances, indices = index.search(text_embeddings, k=topk)
    text_query_as_title = [(txt,) * topk for txt in text_query]
    for Is, Ds, tt in zip(indices, distances, text_query_as_title):
        indices_distances = list(zip(Is, Ds))
        indices_distances.sort(key=lambda x: x[1])  # Sort based on the distances

        if vizualization:
            viz_subplot(indices_distances, image_paths, tt)


if __name__ == "__main__":
    main(using_stt=False)
    main(using_stt=True)
