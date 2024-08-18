"""
Prepare image/text pairs into training/finetuning format for CN-CLIP model

# Example usage:
image_files = glob.glob("path/to/images/fmh-*.jpg")
train_files, valid_files, test_files = split_data(image_files)
images_to_base64(image_files, "train_imgs.tsv", "valid_imgs.tsv", "test_imgs.tsv")
text_image_to_jsonl(image_files, "train_texts.jsonl", topk=8)

"""
#  NOTE run this script as module, e.g., 'py -m src.data.utils.transform_raw_images'

import base64
import glob
import json
import os
import random
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from src.data.faiss_index import create_faiss_index
#  from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer

# Set random seed for reproducibility
random.seed(42)


# 应该使用与CN-CLIP一致的文本端嵌入模型；CN-CLIP：vit-h-14/RoBERTa-wwm-ext-large-chinese
# IDEA 不一定使用与 CN-CLIP 一致的文本端模型，因为主要用于相似图片描述文字的检索，使用一
# 般的 sentence-transformer embedding 模型其实更方便 (2024-08-18 Sun)
# 这里使用 faiss 的 index.search() 会导致相同数量的返回结果（除非做进一步后处理）

# Get the first snapshot directory when using hugggingface hub directory
base_dir = "./models/hfLLMs/RoBERTa-wwm-ext-large-chinese/snapshots"
snapshot_dir = os.path.join(base_dir, os.listdir(base_dir)[0])
EMB_MODEL = BertModel.from_pretrained(snapshot_dir)
EMB_TOKENIZER = BertTokenizer.from_pretrained(snapshot_dir)

output_filepath = Path("../fmhData/datasets/XiaoMuZai/")
input_filepath = output_filepath / "raw_images"


def main():
    # output tsv file:
    fpath_str = input_filepath.joinpath("fmh-*.jpg").as_posix()
    image_files = glob.glob(fpath_str)  # list of filepath
    train_files, valid_files, test_files = split_data(image_files, shuffle=True)

    tr_tsv = output_filepath / "train_imgs.tsv"
    images_to_base64(train_files, tr_tsv)

    vl_tsv = output_filepath / "valid_imgs.tsv"
    images_to_base64(valid_files, vl_tsv)

    te_tsv = output_filepath / "test_imgs.tsv"
    images_to_base64(test_files, te_tsv)

    # output jsonl file:
    tr_jsonl = output_filepath / "train_texts.jsonl"
    text_image_to_jsonl(train_files, tr_jsonl, topk=3)

    vl_jsonl = output_filepath / "valid_texts.jsonl"
    text_image_to_jsonl(valid_files, vl_jsonl, topk=3)

    te_jsonl = output_filepath / "test_texts.jsonl"
    text_image_to_jsonl(test_files, te_jsonl, topk=3)


def split_data(image_files, shuffle=True, train_ratio=0.7, valid_ratio=0.2):
    if shuffle:
        random.shuffle(image_files)
    total = len(image_files)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    return (
        image_files[:train_end],
        image_files[train_end:valid_end],
        image_files[valid_end:],
    )


def images_to_base64(image_files, filename):
    """turn image into byte data"""
    img_id = 0

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

    b64_imgs = process_images(image_files)

    with open(filename, "w") as f:
        f.write("\n".join(b64_imgs))
    print(f"write tsv file to {filename}")


def text_image_to_jsonl(image_files, output_file, topk=10, verbose=False):
    """

    ${split}_texts.jsonl Example:
    {"text_id": 8428, "text": "高级感托特包斜挎", "image_ids": [1076345, 517602]}
    """
    # 私人图库中，图片内容略有不同但可能都是描述同一件事情（睡姿不同都是睡）；
    # 或者用同一个文字描述来命名不同图片内容（比喻、比拟、指桑骂槐等等）。
    # 这里要处理的是：将同义语句的不同图片给对应起来（Example）
    # 如何确定语句之间的“同义”程度？语句向量相似度。

    # 在语句向量中找相似向量，topk threshold=0.8 + distance threshold
    # jsonl: {text_id: idx, text: text, image_ids: 相似语句对应的图片IDs}
    # 因为两者都是从零开始，因此相似检索返回的text_id就是image_id

    # image_files is list of filepath (hence text-infos in filenames)
    texts = get_text_from_filename(image_files)
    text_embeddings = embed_texts(texts)
    index = create_faiss_index(np.array(text_embeddings))
    #  save_faiss_index(index, file_path=img_embs_file_path)

    # format into jsonl entry:
    data = []
    for idx, (text, emb) in enumerate(zip(texts, text_embeddings)):
        emb = np.array([emb])  # Ensure emb is a 2D array or index.search will be failed
        distances, indices = index.search(emb, k=topk)  # emb is the query vector
        if verbose == 2:
            for idx in indices:
                print(texts[idx])
            breakpoint()
        entry = {
            "text_id": idx,
            "text": text,
            "image_ids": indices[0].tolist(),
            # where is the 1.0 come from, the Magic number??? Better question: how to
            # get a better Magic number for filtering neighbors?
            #  "image_ids": [
            #      idx for (idx, dis) in zip(indices[0], distances[0]) if dis < 1.0
            #  ],
        }
        entry = convert_numpy_objects(entry)
        data.append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"write jsonl file to {output_file}")


def get_text_from_filename(image_files):
    return [
        file_name.split("/")[-1].replace("fmh-", "").rsplit(".", 1)[0]
        for file_name in image_files
    ]


def embed_texts(sentence, verbose=False):

    # Tokenize the sentence and convert to input IDs, max_length=52 for CLIP
    inputs = EMB_TOKENIZER(
        sentence, return_tensors="pt", max_length=52, truncation=True, padding=True
    )

    EMB_MODEL.eval()
    # Forward pass, get hidden states
    with torch.no_grad():
        outputs = EMB_MODEL(**inputs)

    # Extract the embeddings (last hidden state)
    # outputs[0] is the sequence of hidden states at the output of the last layer
    sentence_embeddings = outputs.last_hidden_state

    # If you want to average the token embeddings to get a single vector for the sentence
    sentence_embedding = torch.mean(sentence_embeddings, dim=1)

    if verbose == 2:
        print(sentence_embedding)

    return sentence_embedding


def convert_numpy_objects(data):
    if isinstance(data, dict):
        return {k: convert_numpy_objects(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_objects(i) for i in data]
    elif isinstance(data, np.integer):
        return int(data)
    else:
        return data


if __name__ == "__main__":
    main()
