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


# FIXME plot images using subplots for better display
def viz(indices_distances, image_path, titles):
    for (idx, distance), title in zip(indices_distances, titles):
        path = image_path[idx]
        im = Image.open(path)
        plt.title(f"query_text: {title}\nimg_path: {path}")
        plt.imshow(im)
        plt.show()
