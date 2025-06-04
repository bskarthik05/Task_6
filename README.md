# K-Nearest Neighbors (KNN) Classification on Iris Dataset  

This project demonstrates the implementation of the K-Nearest Neighbors (KNN) algorithm for a classification task using the well-known Iris dataset. It covers data loading, preprocessing, model training with different K values, evaluation, and visualization of decision boundaries.

## Project Objective
The main goal is to understand and apply the KNN algorithm to a classification problem, experiment with hyperparameter tuning (specifically the value of K), and evaluate the model's performance using standard metrics.

## Dataset
The [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) is used for this project. It contains 150 samples from three species of Iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Four features were measured from each sample:
* Sepal Length (cm)
* Sepal Width (cm)
* Petal Length (cm)
* Petal Width (cm)

The provided `Iris.csv` file is used as the data source.

## Process Overview
The Python script performs the following steps:

1.  **Load Data**:
    * The `Iris.csv` dataset is loaded using Pandas.
    * The `Id` column, if present, is dropped as it's not relevant for classification.

2.  **Data Exploration & Preparation**:
    * Basic information about the dataset (first few rows, data types) is printed.
    * Features (X) are separated from the target variable (y - 'Species').

3.  **Train-Test Split**:
    * The dataset is split into a training set (70%) and a testing set (30%).
    * Stratification (`stratify=y`) is used to ensure that the class proportions are maintained in both the training and testing splits.

4.  **Feature Scaling**:
    * Features are normalized using `StandardScaler` from Scikit-learn. This standardizes features by removing the mean and scaling to unit variance, which is crucial for distance-based algorithms like KNN.

5.  **KNN Implementation & K Value Experimentation**:
    * The `KNeighborsClassifier` from Scikit-learn is used.
    * The model is trained and evaluated for a range of K values (1 to 25 in this script).
    * Accuracy for each K value is calculated and printed.
    * A plot of **Accuracy vs. K Value** is generated and saved as `accuracy_vs_k.png` to help visualize the optimal K.

6.  **Model Evaluation**:
    * The K value that yields the highest accuracy on the test set is selected as the "best K".
    * A final KNN model is trained using this best K.
    * The final model's performance is evaluated using:
        * **Accuracy Score**: The proportion of correctly classified instances.
        * **Confusion Matrix**: A table showing the performance of the classification model on the test data for which the true values are known.

7.  **Decision Boundary Visualization**:
    * To understand how the KNN model separates the classes, decision boundaries are plotted.
    * For simplicity and 2D visualization, only the first two scaled features (e.g., Scaled Sepal Length and Scaled Sepal Width) are used for this plot.
    * The decision boundaries plot is saved as `knn_decision_boundaries.png`.

## Libraries Used
* **Pandas**: For data manipulation and loading CSV files.
* **Scikit-learn (sklearn)**: For machine learning tasks including:
    * `train_test_split`: Splitting data.
    * `StandardScaler`: Feature scaling.
    * `KNeighborsClassifier`: KNN algorithm.
    * `accuracy_score`, `confusion_matrix`: Model evaluation metrics.
* **Matplotlib**: For plotting graphs (accuracy vs. K, decision boundaries).
* **NumPy**: For numerical operations, especially for creating the meshgrid for decision boundaries.

## Outputs
The script will:
* Print the initial dataset information, shape of train/test sets, and scaled data examples.
* Print the accuracy for each K value tested.
* Generate and save a plot `accuracy_vs_k.png`.
* Print the best K found and the corresponding accuracy and confusion matrix.
* Generate and save a plot `knn_decision_boundaries.png` visualizing the decision boundaries for the first two features.

## Key Learnings from the Task
* **Instance-based Learning**: Understanding how KNN makes predictions based on the majority class among its 'K' closest neighbors in the feature space.
* **Euclidean Distance**: Recognizing its role (as the default in Scikit-learn's KNN) in measuring similarity or dissimilarity between data points.
* **Importance of K Selection**: Observing how the choice of K impacts model performance and the methods to select an optimal K (e.g., iterating through values and checking accuracy).
* **Model Evaluation Techniques**: Using accuracy and confusion matrices to assess the classifier's effectiveness.
* **Decision Boundary Visualization**: Gaining insight into how the model partitions the feature space to classify different instances.
