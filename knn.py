class AnnoyKNeighborsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, distance='euclidean', weights='uniform'):
        self.distance = distance
        self.trees = 50
        self.weights = weights
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        self.y = y.values
        f = X.shape[1]
        self.t = AnnoyIndex(f, self.distance)  # Length of item vector that will be indexed
        for i, v in enumerate(X):
            self.t.add_item(i, v)

        self.t.build(self.trees)
        
    
    def predict(self, X):
        y_pred = []
        for x in X:
            preds, dists = self.t.get_nns_by_vector(x, self.n_neighbors, include_distances=True)
            dists = np.array(dists)
            preds = self.y[preds]
            if self.weights == 'distance':
                pred = np.average(preds, weights=1 - dists / dists.sum())
            else:
                pred =preds.mean()
            y_pred.append(pred)
        return np.array(y_pred)
		