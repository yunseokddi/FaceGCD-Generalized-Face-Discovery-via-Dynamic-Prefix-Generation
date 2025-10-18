import numpy as np
import os
import torch
import faiss

if __name__ == "__main__":
    feature_path = os.path.join(
        '/home/compu/yunseok/clone_src/AIGCD/experiment_src/arcface/results/arcface_youtubefaces_1000_GCD_fine_tuning_cluster_train/npy_test',
        'feature.npy')
    knn_path = os.path.join(
        '/home/compu/yunseok/clone_src/AIGCD/experiment_src/arcface/results/arcface_youtubefaces_1000_GCD_fine_tuning_cluster_train/npy_test',
        'knn_graph.npy')

    features = np.load(feature_path)

    '''
    features : (47956, 512)
    knn : (47956, 1001)
    '''

    d = 512
    index = faiss.IndexFlatL2(d)
    index.add(features)
    k = 1001

    D, I = index.search(features, k)

    np.save(knn_path, I)


    print(I.shape)
