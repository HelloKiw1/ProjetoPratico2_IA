import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        # Calcular a probabilidade para cada classe
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.prior = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.prior[c] = len(X_c) / len(X)

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.prior[c])
            likelihood = np.sum(np.log(self._pdf(c, x)))
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, c, x):
        mean = self.mean[c]
        var = self.var[c]
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean) ** 2 / (2 * var))
