from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from collections import Counter
import numpy as np

def loocv_knn(X: np.ndarray, y: np.ndarray, k_list=[1,3,5,7]):
    """
    Evaluate KNN for each k in k_list using leave-one-out CV.
    Prints top-1 and top-k accuracy.
    """
    loo = LeaveOneOut()
    results = {k: {"hit1":0, "hitk":0} for k in k_list}
    n = len(X)

    for train_idx, test_idx in loo.split(X):
        # pipeline: scale → PCA(30 components)
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("pca",   PCA(n_components=30, random_state=42))
        ])
        X_tr = pipe.fit_transform(X[train_idx])
        X_te = pipe.transform(X[test_idx])

        for k in k_list:
            knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
            knn.fit(X_tr, y[train_idx])
            neigh_idx = knn.kneighbors(X_te, return_distance=False)[0]
            neigh_labels = y[train_idx][neigh_idx]
            true_label = y[test_idx][0]

            # majority-vote prediction
            pred = Counter(neigh_labels).most_common(1)[0][0]
            results[k]["hit1"] += int(pred == true_label)
            results[k]["hitk"] += int(true_label in neigh_labels)

    for k in k_list:
        print(f"K={k} | top-1 {results[k]['hit1']/n:.3f} | top-{k} {results[k]['hitk']/n:.3f}")
