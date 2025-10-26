from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto").fit(X)
print("Clusters:", kmeans.labels_[:10])

# Classificação
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = DecisionTreeClassifier().fit(X_train, y_train)
print("Acurácia:", clf.score(X_test, y_test))

# Anomalias
anom = IsolationForest(contamination=0.1).fit(X)
print("Anomalias detectadas:", list(anom.predict(X)).count(-1))
