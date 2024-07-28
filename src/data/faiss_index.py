import faiss


def create_faiss_index(embeddings, metric=faiss.METRIC_L2):
    """
    Create a FAISS index and add embeddings to it.
    :param embeddings: The image embeddings as a NumPy array
    :param metric: Distance metric for FAISS (default: L2)
    :return: The FAISS index
    """
    dimension = embeddings.shape[1]

    if metric == faiss.METRIC_L2:
        index = faiss.IndexFlatL2(dimension)
    elif metric == faiss.METRIC_INNER_PRODUCT:
        index = faiss.IndexFlatIP(dimension)
    else:
        raise ValueError(
            "Unsupported metric. Use faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT."
        )

    index.add(embeddings)
    return index


def save_faiss_index(index, file_path):
    """
    Save the FAISS index to a file.
    :param index: The FAISS index to be saved
    :param file_path: Path to save the index
    """
    faiss.write_index(index, file_path)
    print(f"Saved index to {file_path}")


def load_faiss_index(file_path):
    """
    Load a FAISS index from a file.
    :param file_path: Path to the FAISS index file
    :return: Loaded FAISS index
    """
    return faiss.read_index(file_path)
