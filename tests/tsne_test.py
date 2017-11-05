from tsne_plot import *

if __name__ == '__main__':
    # sampling_ratio = 0.79
    seed = 42
    perplexity = 25
    learning_rate = 20
    n_iter = 250
    n_split = 10
    # dataset = "fraud"
    # X, y = load_fraud_detection()
    # print(Counter(y))
    dataset = "santander"
    X, y = load_santander()

    n_samples = len(y)
    n_features = X.shape[1]
    random.seed(seed)

    sampled_X, sampled_y = reduce_majority(X, y, gamma=0.9)
    sampling_ratio = len(sampled_y) / len(y)
    c = Counter(sampled_y)
    print(c)

    cv_tsne_result = compute_tsne_cv(sampled_X, sampled_y, n_split=n_split, random_state=seed, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
    # print(cv_tsne_result.shape, sampled_X.shape, sampled_y.shape)

    all_result = np.concatenate((sampled_X, sampled_y[:, np.newaxis], cv_tsne_result), axis=1)
    np.save("cv-tsne-result-{}-{:.4f}-{}-{}.npy".format(dataset, sampling_ratio, c[0], c[1]), all_result)

    # t_SNE_plot(X, y, id="fraud", perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)