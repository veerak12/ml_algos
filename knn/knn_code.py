import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

def load_data(file_path):
    """Load dataset from file"""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the dataset"""
    X = df.drop("isfraud", axis=1).values
    y = df["isfraud"].values
    return X, y

def train_initial_knn(X_train, y_train):
    """Train initial k-NN classifier with 3 neighbors"""
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y_train)
    return knn_classifier

def find_optimal_k(X_train, y_train):
    """Find the optimal number of neighbors using cross-validation"""
    grid = {'n_neighbors': np.arange(1, 25)}
    knn_classifier = KNeighborsClassifier()
    knn = GridSearchCV(knn_classifier, grid, cv=10)
    knn.fit(X_train, y_train)
    return knn.best_params_, knn.best_score_

def visualize_accuracy(k_range, k_scores, score_scaled):
    """Visualize accuracy for different values of k"""
    plt.figure(figsize=(12, 6))
    plt.plot(k_range, k_scores, marker='o')
    plt.title('k-NN Varying number of neighbors')
    plt.xlabel('Number of neighbors (k)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

def scale_and_train(X_train, y_train):
    """Scale the data and train the k-NN classifier"""
    pipeline_order = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=1))]
    pipeline = Pipeline(pipeline_order)
    knn_classifier_scaled = pipeline.fit(X_train, y_train)
    return knn_classifier_scaled

def visualize_performance(k_range, k_scores, score_scaled):
    """Visualize the performance before and after scaling"""
    plt.figure(figsize=(12, 6))
    plt.plot(k_range, k_scores, marker='o', label='Before Scaling')
    plt.axhline(y=score_scaled, color='r', linestyle='--', label='After Scaling (n_neighbors=1)')
    plt.title('k-NN Performance Before and After Scaling')
    plt.xlabel('Number of neighbors (k)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Load data
    df = load_data("fraud_prediction.csv")
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Split the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train initial k-NN classifier
    knn_classifier = train_initial_knn(X_train, y_train)
    
    # Extract the accuracy score from the test sets
    score = knn_classifier.score(X_test, y_test)
    print(f"Initial k=3 score: {score*100:.2f}%")
    
    # Find the optimal number of neighbors
    best_params, best_score = find_optimal_k(X_train, y_train)
    print(f"Optimal number of neighbors: {best_params['n_neighbors']}")
    print(f"Best cross-validated score: {best_score*100:.2f}%")
    
    # Visualize accuracy for different values of k
    k_range = np.arange(1, 25)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        k_scores.append(knn.score(X_test, y_test))
    visualize_accuracy(k_range, k_scores, score)
    
    # Scale the data and train the k-NN classifier
    knn_classifier_scaled = scale_and_train(X_train, y_train)
    
    # Extract the score
    score_scaled = knn_classifier_scaled.score(X_test, y_test)
    print(f"k-NN classified score after scaling: {score_scaled*100:.2f}%")
    
    # Visualize the performance before and after scaling
    visualize_performance(k_range, k_scores, score_scaled)

if __name__ == "__main__":
    main()
