"""
Plot the semantic search result
"""

from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt

# 中文减号显示问题
mpl.rcParams['axes.unicode_minus'] = False
# 中文汉字显示问题
# 注这里指定参数的时候，名字与ttf文件的名字不同: simfang.ttf >> 'FangSong'
mpl.rcParams['font.sans-serif'] = ['FangSong']


def viz(indices_distances, image_path, titles):
    for (idx, distance), title in zip(indices_distances, titles):
        path = image_path[idx]
        im = Image.open(path)
        plt.title(f"query_text: {title}\nimg_path: {path}")
        plt.imshow(im)
        plt.show()


def viz_subplot(indices_distances, image_path, titles, cols=3, figsize=(15, 10)):
    """
    Visualize images with their corresponding indices and distances using subplots.

    Args:
        indices_distances (list of tuples): List of (index, distance) tuples for the images.
        image_path (list of str): List of image paths.
        titles (list of str): List of titles for each image.
        cols (int): Number of columns in the subplot grid.
        figsize (tuple): Size of the figure.
    """
    num_images = len(indices_distances)
    rows = (num_images + cols - 1) // cols  # Calculate number of rows needed

    plt.figure(figsize=figsize)

    for i, ((idx, distance), title) in enumerate(zip(indices_distances, titles)):
        path = image_path[idx]
        im = Image.open(path)

        plt.subplot(rows, cols, i + 1)  # Add a subplot
        plt.imshow(im)
        plt.title(f"img_path: {path}")
        plt.axis('off')  # Hide axis

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(f"query_text: {title}", fontsize=16)
    plt.subplots_adjust(top=0.9)  # Adjust top spacing to make room for the title
    plt.show()
