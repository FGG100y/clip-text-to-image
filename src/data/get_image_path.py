import itertools
from pathlib import Path


def read(img_dir="data/raw/weixin_img/", patterns=("*.jpg", "*.png"), as_posix=False):
    """Get image's filepath [default from data/raw/weixin_img/]"""
    data_path = Path(img_dir)
    image_path = list(
        itertools.chain.from_iterable(data_path.glob(pattern) for pattern in patterns)
    )

    if as_posix:
        return [imgp.as_posix() for imgp in image_path]

    return image_path
