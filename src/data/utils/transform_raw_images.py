"""
# Example usage:
image_files = glob.glob("path/to/images/fmh-*.jpg")
train_files, valid_files, test_files = split_data(image_files)
images_to_base64(image_files, "train_imgs.tsv", "valid_imgs.tsv", "test_imgs.tsv")
text_image_to_jsonl(train_files, "train_texts.jsonl", threshold=0.8)
text_image_to_jsonl(valid_files, "valid_texts.jsonl", threshold=0.8)
text_image_to_jsonl(test_files, "test_texts.jsonl", threshold=0.8)


"""

from PIL import Image
from io import BytesIO
import base64
import json
from pathlib import Path
import glob
import random

#  import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Set random seed for reproducibility
random.seed(42)

output_filepath = Path("../fmhData/datasets/XiaoMuZai/")
input_filepath = output_filepath / "raw_images"
# FIXME 应该使用与CN-CLIP一致的文本端嵌入模型
emb_model_path = "./models/hfLLMs/bge-large-zh-v1.5"
EMBEDDING_MODEL = SentenceTransformer(emb_model_path)


def main():
    # output tsv file:
    fpath_str = input_filepath.joinpath("fmh-*.jpg").as_posix()
    image_files = glob.glob(fpath_str)

    tr_tsv = output_filepath / "train_imgs.tsv"
    vl_tsv = output_filepath / "valid_imgs.tsv"
    te_tsv = output_filepath / "test_imgs.tsv"
    images_to_base64(image_files, tr_tsv, vl_tsv, te_tsv)

    # output jsonl file:
    tr_jsonl = output_filepath / "train_texts.jsonl"
    vl_jsonl = output_filepath / "valid_texts.jsonl"
    te_jsonl = output_filepath / "test_texts.jsonl"
    train_files, valid_files, test_files = split_data(image_files)
    text_image_to_jsonl(train_files, tr_jsonl, threshold=0.8)
    text_image_to_jsonl(valid_files, vl_jsonl, threshold=0.8)
    text_image_to_jsonl(test_files, te_jsonl, threshold=0.8)


def embed_texts(texts):
    return EMBEDDING_MODEL.encode(texts)


def split_data(image_files, train_ratio=0.7, valid_ratio=0.2):
    random.shuffle(image_files)
    total = len(image_files)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    return (
        image_files[:train_end],
        image_files[train_end:valid_end],
        image_files[valid_end:],
    )


def images_to_base64(image_files, train_file, valid_file, test_file):
    img_id = 100000
    train_files, valid_files, test_files = split_data(image_files)

    def process_images(files):
        data = []
        nonlocal img_id
        for file_name in files:
            img = Image.open(file_name)
            img_buffer = BytesIO()
            img.save(img_buffer, format=img.format)
            byte_data = img_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            data.append(f"{img_id}\t{base64_str}")
            img_id += 1
        return data

    train_data = process_images(train_files)
    valid_data = process_images(valid_files)
    test_data = process_images(test_files)

    with open(train_file, "w") as f:
        f.write("\n".join(train_data))
    print(f"write file to {train_file}")

    with open(valid_file, "w") as f:
        f.write("\n".join(valid_data))
    print(f"write file to {valid_file}")

    with open(test_file, "w") as f:
        f.write("\n".join(test_data))
    print(f"write file to {test_file}")


def text_image_to_jsonl(image_files, output_file, threshold=0.8):
    img_id = 100000
    text_id = 1000
    text_to_id = {}
    text_to_images = {}

    text_infos = [
        file_name.split("/")[-1].replace("fmh-", "").rsplit(".", 1)[0]
        for file_name in image_files
    ]
    embeddings = embed_texts(text_infos)

    for idx, text_info in enumerate(text_infos):
        text_embedding = embeddings[idx]

        matched_text = None
        max_similarity = 0

        for existing_text, tid in text_to_id.items():
            existing_embedding = embed_texts([existing_text])[0]
            similarity = cosine_similarity([text_embedding], [existing_embedding])[0][0]
            if similarity > max_similarity and similarity >= threshold:
                max_similarity = similarity
                matched_text = existing_text

        if matched_text:
            #  text_id = text_to_id[matched_text]
            text_to_images[matched_text].append(img_id)
        else:
            text_to_id[text_info] = text_id
            text_to_images[text_info] = [img_id]
            text_id += 1

        img_id += 1

    data = []
    for text_info, tid in text_to_id.items():
        entry = {
            "text_id": tid,
            "text": text_info,
            "image_ids": text_to_images[text_info],
        }
        data.append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"write file to {output_file}")


if __name__ == "__main__":
    main()
