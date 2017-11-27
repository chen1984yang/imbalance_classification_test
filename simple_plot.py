import time
from sklearn import manifold
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np
import matplotlib.patches as mpatches


def compute_tsne(X, perplexity, learning_rate, n_iter):
    start = time.time()
    print("start tsne", len(X))
    tsne = manifold.TSNE(
        n_components=2,
        init='pca',
        random_state=0,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter
    )
    X_tsne = tsne.fit_transform(X)
    print("finish tsne", time.time() - start)
    return X_tsne

# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], ".",
                 color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)



from collections import Counter
from dataset import *
import random
from sampling import *
from sklearn.model_selection import train_test_split, StratifiedKFold


def compute_tsne_cv(X, y, n_split=10, perplexity=5, learning_rate=200, n_iter=1000, random_state=42):
    cv = StratifiedKFold(n_splits=n_split, random_state=random_state)
    tsne_result = np.zeros(shape=(len(y), 3 * n_split))
    for i, t in zip(range(n_split), cv.split(X, y)):
        train_index, valid_index = t
        train_X = X[train_index]
        train_y = y[train_index]
        valid_X = X[valid_index]
        valid_y = y[valid_index]

        train_X_tsne = compute_tsne(train_X, perplexity, learning_rate, n_iter)
        tsne_result[train_index, i * 3: i * 3 + 2] = train_X_tsne
        tsne_result[train_index, i * 3 + 2] = 0

        valid_X_tsne = compute_tsne(valid_X, perplexity, learning_rate, n_iter)
        tsne_result[valid_index, i * 3: i * 3 + 2] = valid_X_tsne
        tsne_result[valid_index, i * 3 + 2] = 1
    return tsne_result

def parse_cv_tsne_result(result, n_split=10):
    print(result.shape)
    tsne_result = result[:, - n_split * 3:]
    tsne_list = []
    for i in range(n_split):
        tsne_list.append((tsne_result[:, i * 3 : i * 3 + 2], tsne_result[:, i * 3 + 2]))
    label = result[:, - n_split * 3 - 1]
    original_feature = result[:, :- n_split * 3 - 1]
    # print(original_feature.shape, label.shape, tsne_result.shape)
    return original_feature, label, tsne_list


def save_plot2png(x, y, color, path, title="", color_map=None):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, )  # create figure & 1 axis
    # ax.plot([0, 1, 2], [10, 20, 3])
    if type(color) == np.ndarray:
        color = color.tolist()
    ax.scatter(x, y, color=color, s=0.5)
    
    if color_map is None:
        color_map = {}
        
    patches = []

    for legend_name in color_map:

        patch = mpatches.Patch(color=color_map[legend_name], label=legend_name)
        patches.append(patch)
    plt.legend(handles=patches)
    plt.title(title)
    plt.savefig(path, dpi=300)  # save the figure to file
    plt.close(fig)


def save_linechart(series, x_values, path, title="", x_title="", y_title="", color_map=None):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.grid(linestyle='dotted')
    idx = 0
    handles = []
    series_names = []
    
    if color_map is None:
        color_map = {}

    reference_series = "Dummy"
    legends = sorted(series.keys())
    if reference_series in legends:
        legends.remove(reference_series)
        legends = [reference_series] + legends
    linewidth = 1.0
    for key in legends:
        series_names.append(key)
        color = None
        if key in color_map:
            color = color_map[key]
        # print(key, color)
        w = linewidth
        if key == reference_series:
            w *= 4
        handle, = ax.plot(x_values, series[key], label=key, color=color, linewidth=w)
        handles.append(handle)
        
    plt.legend(handles, series_names)
    
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig(path, dpi=300)
    print("plot", title, "saved to", path)
    plt.close(fig)


if __name__ == '__main__':
    N = 100
    x = np.random.rand(N)
    y = np.random.rand(N)
    x_values = np.arange(N)
    color_map = {
        "x": "#3c3c3c"
    }
    save_linechart({
        "x": x,
        "y": y
    }, x_values, "linechart.png", "demo", "x", "y", color_map=color_map)
    # colors = np.random.rand(N)
    # save_plot(x, y, colors, 'to1.png')