from naive_bayes import NaiveBayes
from kmeans import KMeans
from preprocess import load_data, train_test_split, normalize_data

def run_experiment():
    data = load_data('data/iris.csv')
    X = data.iloc[:, :-1].values  # Características
    y = data.iloc[:, -1].values  # Rótulos

    X = normalize_data(X)  # Normalização dos dados

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Aplicar Naive Bayes
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    # Calcular a acurácia do Naive Bayes
    accuracy = np.mean(y_pred == y_test)
    print(f'Acurácia do Naive Bayes: {accuracy * 100:.2f}%')

    # Clusterização com KMeans
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_train)
    clusters = kmeans.labels

    print("Clusters gerados pelo KMeans:", clusters)

if __name__ == "__main__":
    run_experiment()
