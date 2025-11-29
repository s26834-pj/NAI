"""
Machine Learning Project
===================================

Authors: Wiktor Rapacz, Hanna Paczoska

Solution overview
-----------------
This project compares the performance of two classification algorithms:

    * Decision Tree (DecisionTreeClassifier)
    * Support Vector Machine (SVM, sklearn.svm.SVC)

on two different datasets:

1. Wine Quality (Red Wine)
   - Goal: classify the quality of red wine based on its physicochemical properties.

2. Cats Dataset (Kaggle)
   - Goal: classify the gender of a cat (Male / Female) based on simple features
     such as breed, age, weight and fur color.
   - Online source (Kaggle, required in project description):
       https://www.kaggle.com/datasets/waqi786/cats-dataset?resource=download

Program workflow
----------------
For each dataset the program:

    1. Loads the data from a CSV file.
    2. Performs basic visualisation of the data:
       - Wine: histograms of all numeric features and correlation heatmap.
       - Cats: histograms of numeric features and scatter plot
         (Age vs Weight coloured by gender).
    3. Splits the data into training and test sets.
    4. Trains a Decision Tree classifier and evaluates it using:
       accuracy, classification report and confusion matrix.
    5. Trains an SVM classifier (with RBF kernel) and evaluates it in the same way.
    6. Shows example predictions for a few manually constructed / selected samples.
    7. Demonstrates how different SVM kernels and hyperparameters (kernel, C, gamma)
       influence the classification quality by training and evaluating several variants.

Environment and setup instructions
----------------------------------
1. Recommended Python version:
       Python 3.10+ (tested with a modern CPython 3.x)

2. Required Python packages (install inside your virtual environment):

       pip install numpy pandas scikit-learn matplotlib seaborn

3. Project directory structure (example):

       Zadanie_4/
       ├── main.py
       └── resources/
           ├── winequality-red.csv
           └── cats_dataset.csv

4. How to run the project:
       python main.py

   During execution several plots will be displayed. The program continues
   after closing each figure window.

"""
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
#     HELPER FUNCTIONS
# =========================

def load_dataset(file_path, columns=None):
    """
    Load a dataset from a local file.

    Parameters:
    file_path (str): Path to the dataset file.
    columns (list, optional): List of column names to assign to the dataset.

    Returns:
    DataFrame: A pandas DataFrame containing the dataset.
    """
    return pd.read_csv(file_path, names=columns) if columns else pd.read_csv(file_path)


def prepare_features_and_target(dataframe, target_label):
    """
    Split a dataset into features and target variables.

    Parameters:
    dataframe (DataFrame): The input dataset as a pandas DataFrame.
    target_label (str): The name of the column to be used as the target variable.

    Returns:
    Tuple[DataFrame, Series]: A tuple containing the feature set (X) and target variable (y).
    """
    features = dataframe.drop(columns=[target_label])
    target = dataframe[target_label]
    return features, target


def train_decision_tree_model(features_train, target_train, **kwargs):
    """
    Train a Decision Tree model on the given data.

    Parameters:
    features_train (DataFrame): Training features.
    target_train (Series): Training target labels.
    kwargs: Additional parameters for DecisionTreeClassifier.

    Returns:
    DecisionTreeClassifier: A trained Decision Tree classifier.
    """
    model = DecisionTreeClassifier(**kwargs)
    model.fit(features_train, target_train)
    return model


def train_svm_model(features_train, target_train, kernel="rbf", c_value=1.0, gamma="scale"):
    """
    Train a Support Vector Machine (SVM) model on the given data.

    Parameters:
    features_train (DataFrame): Training features.
    target_train (Series): Training target labels.
    kernel (str): Kernel function ('linear', 'poly', 'rbf', 'sigmoid').
    c_value (float): Regularization parameter (C).
    gamma (str or float): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

    Returns:
    Pipeline: A sklearn Pipeline containing StandardScaler and SVC.
    """
    # SVM działa lepiej po standaryzacji cech
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", svm.SVC(kernel=kernel, C=c_value, gamma=gamma))
    ])
    model.fit(features_train, target_train)
    return model


def assess_model_performance(model, features_test, target_test, title="Model performance"):
    """
    Evaluate the performance of a trained model.

    Parameters:
    model: The trained machine learning model (can be Pipeline).
    features_test (DataFrame): Testing features.
    target_test (Series): True labels for the testing data.
    title (str): Title printed before metrics.

    Returns:
    None
    """
    print(f"\n=== {title} ===")
    predictions = model.predict(features_test)
    print("Accuracy:", accuracy_score(target_test, predictions))
    print(
        "\nClassification Report:\n",
        classification_report(target_test, predictions, zero_division=0)
    )
    print("Confusion Matrix:\n", confusion_matrix(target_test, predictions))


def plot_dataset_histograms(dataframe, title="Histograms", numeric_only=True):
    """
    Plot histograms for numeric columns in a dataset.

    Parameters:
    dataframe (DataFrame): The dataset to visualize.
    title (str): Title of the plot.
    numeric_only (bool): If True, only numeric columns will be plotted.

    Returns:
    None
    """
    if numeric_only:
        dataframe = dataframe.select_dtypes(include=[np.number])
    if dataframe.empty:
        print("No numeric data available for visualization.")
    else:
        dataframe.hist(bins=16, figsize=(15, 10))
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


def plot_correlation_heatmap(dataframe, title="Correlation heatmap"):
    """
    Plot a correlation heatmap for numeric features.

    Parameters:
    dataframe (DataFrame): Dataset with numeric features.
    title (str): Title of the heatmap.

    Returns:
    None
    """
    numeric_df = dataframe.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("No numeric data available for correlation heatmap.")
        return

    plt.figure(figsize=(10, 8))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def demonstrate_svm_kernels(
    x_train,
    x_test,
    y_train,
    y_test,
    kernels_params,
    dataset_name="dataset"
):
    """
    Train and evaluate SVM models with different kernels and parameters.

    Parameters:
    x_train, x_test, y_train, y_test: Train/test splits.
    kernels_params (list of dict): Each dict contains kernel, C, gamma, and optional name.
    dataset_name (str): Name of dataset for printing.

    Returns:
    None
    """
    print(f"\n### SVM kernel comparison on {dataset_name} ###")
    for params in kernels_params:
        kernel = params.get("kernel", "rbf")
        c_value = params.get("C", params.get("c_value", 1.0))
        gamma = params.get("gamma", "scale")
        name = params.get("name", f"SVM (kernel={kernel}, C={c_value}, gamma={gamma})")

        svm_model = train_svm_model(
            x_train,
            y_train,
            kernel=kernel,
            c_value=c_value,
            gamma=gamma
        )
        assess_model_performance(
            svm_model,
            x_test,
            y_test,
            title=f"{name} on {dataset_name}"
        )


def predict_example(model, example_data, feature_names, description="Example prediction"):
    """
    Use a trained model to predict class for sample input.

    Parameters:
    model: Trained model (DecisionTree or SVM Pipeline).
    example_data (list or array-like): Single sample or list of samples.
    feature_names (list): Names of the features in correct order.
    description (str): Description printed before prediction.

    Returns:
    None
    """
    print(f"\n--- {description} ---")
    example_df = pd.DataFrame(example_data, columns=feature_names)
    preds = model.predict(example_df)
    print("Input data:")
    print(example_df)
    print("Predicted class/label:", preds)


# =========================
#     MAIN WORKFLOW
# =========================

def main():
    """
    Main execution function.

    1. Wine Quality dataset (red wine) - classification of wine quality.
    2. Cats dataset (Kaggle) - classification of cat gender.
    """

    # ---------- 1) WINE QUALITY DATASET ----------
    wine_dataset_path = "resources/winequality-red.csv"
    # W zbiorze winequality-red.csv kolumna 'quality' jest celem (etykietą).
    wine_data = pd.read_csv(wine_dataset_path, sep=';')

    print("=== Processing Wine Quality Dataset (Red Wine) ===")
    print("Dataset shape:", wine_data.shape)
    print(wine_data.head())

    # Przykładowa wizualizacja: histogramy + mapa korelacji
    plot_dataset_histograms(wine_data, title="Wine dataset - feature distributions")
    plot_correlation_heatmap(wine_data, title="Wine dataset - correlation heatmap")

    wine_target_column = "quality"
    wine_features, wine_target = prepare_features_and_target(wine_data, wine_target_column)

    # Podział na train/test
    x_train_wine, x_test_wine, y_train_wine, y_test_wine = train_test_split(
        wine_features,
        wine_target,
        test_size=0.3,
        random_state=45,
        stratify=wine_target
    )

    # --- Drzewo decyzyjne na winie ---
    wine_tree_model = train_decision_tree_model(
        x_train_wine,
        y_train_wine,
        max_depth=None,
        random_state=45
    )
    assess_model_performance(
        wine_tree_model,
        x_test_wine,
        y_test_wine,
        title="Decision Tree on Wine Quality"
    )

    # --- SVM na winie (domyślne parametry) ---
    wine_svm_model = train_svm_model(
        x_train_wine,
        y_train_wine,
        kernel="rbf",
        c_value=1.0,
        gamma="scale"
    )
    assess_model_performance(
        wine_svm_model,
        x_test_wine,
        y_test_wine,
        title="SVM (RBF kernel) on Wine Quality"
    )

    # --- Przykładowe dane wejściowe dla wina ---
    wine_feature_names = list(wine_features.columns)
    # Weźmy średnią ze zbioru jako "typowe" wino oraz to wino + mała zmiana
    typical_wine = wine_features.mean().values
    slightly_stronger_wine = typical_wine.copy()
    # Załóżmy, że ostatnia cecha to 'alcohol'
    slightly_stronger_wine[-1] += 1.0

    example_wine_samples = [typical_wine, slightly_stronger_wine]
    predict_example(
        wine_tree_model,
        example_wine_samples,
        wine_feature_names,
        description="Decision Tree - example predictions for wine"
    )
    predict_example(
        wine_svm_model,
        example_wine_samples,
        wine_feature_names,
        description="SVM - example predictions for wine"
    )

    # --- Demonstracja różnych kernel function dla SVM na winie ---
    wine_kernels_params = [
        {"kernel": "linear", "C": 1.0, "gamma": "scale", "name": "SVM linear C=1.0"},
        {"kernel": "rbf", "C": 1.0, "gamma": "scale", "name": "SVM RBF C=1.0 gamma=scale"},
        {"kernel": "rbf", "C": 10.0, "gamma": 0.1, "name": "SVM RBF C=10 gamma=0.1"},
        {"kernel": "poly", "C": 1.0, "gamma": "scale", "name": "SVM poly (degree=3, default)"},
        {"kernel": "sigmoid", "C": 1.0, "gamma": "scale", "name": "SVM sigmoid C=1.0"},
    ]
    demonstrate_svm_kernels(
        x_train_wine,
        x_test_wine,
        y_train_wine,
        y_test_wine,
        kernels_params=wine_kernels_params,
        dataset_name="Wine Quality (red)"
    )

    # ---------- 2) CATS DATASET (KAGGLE) ----------
    cats_dataset_path = "resources/cats_dataset.csv"
    # Struktura zbioru z Kaggle:
    # https://www.kaggle.com/datasets/waqi786/cats-dataset?resource=download
    cats_data = load_dataset(cats_dataset_path)
    print("\n=== Processing Cats Dataset ===")
    print("Dataset shape:", cats_data.shape)
    print(cats_data.head())

    # Tutaj jasno wybieramy cel klasyfikacji:
    cats_target_column = "Gender"

    # Usuń ewentualne kolumny identyfikatorów, jeśli istnieją
    for id_col in ["CatID", "ID", "Index"]:
        if id_col in cats_data.columns:
            cats_data = cats_data.drop(columns=[id_col])

    # Prosta wizualizacja: histogramy oraz wykres rozrzutu dwóch cech (jeśli są numeryczne)
    plot_dataset_histograms(cats_data, title="Cats dataset - feature distributions")

    numeric_cats = cats_data.select_dtypes(include=[np.number])
    if numeric_cats.shape[1] >= 2:
        plt.figure(figsize=(6, 5))
        x_col = numeric_cats.columns[0]
        y_col = numeric_cats.columns[1]
        sns.scatterplot(
            data=cats_data,
            x=x_col,
            y=y_col,
            hue=cats_target_column
        )
        plt.title(f"Cats dataset - example scatter plot ({x_col} vs {y_col})")
        plt.tight_layout()
        plt.show()

    cats_features, cats_target = prepare_features_and_target(
        cats_data,
        cats_target_column
    )

    # Jeśli są cechy nienumeryczne (np. kolor jako string), prosta enkodacja one-hot:
    cats_features = pd.get_dummies(cats_features)

    x_train_cats, x_test_cats, y_train_cats, y_test_cats = train_test_split(
        cats_features,
        cats_target,
        test_size=0.3,
        random_state=45,
        stratify=cats_target
    )

    # --- Drzewo decyzyjne na kotach ---
    cats_tree_model = train_decision_tree_model(
        x_train_cats,
        y_train_cats,
        max_depth=None,
        random_state=45
    )
    assess_model_performance(
        cats_tree_model,
        x_test_cats,
        y_test_cats,
        title="Decision Tree on Cats Dataset"
    )

    # --- SVM na kotach (RBF) ---
    cats_svm_model = train_svm_model(
        x_train_cats,
        y_train_cats,
        kernel="rbf",
        c_value=1.0,
        gamma="scale"
    )
    assess_model_performance(
        cats_svm_model,
        x_test_cats,
        y_test_cats,
        title="SVM (RBF kernel) on Cats Dataset"
    )

    # --- Przykładowe dane wejściowe dla kotów ---
    cats_feature_names = list(cats_features.columns)
    # Weźmy dwa pierwsze rekordy jako przykładowe dane wejściowe
    example_cat_samples = x_test_cats.iloc[:2].values

    predict_example(
        cats_tree_model,
        example_cat_samples,
        cats_feature_names,
        description="Decision Tree - example predictions for cats"
    )
    predict_example(
        cats_svm_model,
        example_cat_samples,
        cats_feature_names,
        description="SVM - example predictions for cats"
    )

    # --- Demonstracja różnych kernel function dla SVM na kotach ---
    cats_kernels_params = [
        {"kernel": "linear", "C": 1.0, "gamma": "scale", "name": "SVM linear C=1.0"},
        {"kernel": "rbf", "C": 1.0, "gamma": "scale", "name": "SVM RBF C=1.0"},
        {"kernel": "rbf", "C": 5.0, "gamma": 0.5, "name": "SVM RBF C=5 gamma=0.5"},
        {"kernel": "poly", "C": 1.0, "gamma": "scale", "name": "SVM poly (degree=3)"},
    ]
    demonstrate_svm_kernels(
        x_train_cats,
        x_test_cats,
        y_train_cats,
        y_test_cats,
        kernels_params=cats_kernels_params,
        dataset_name="Cats Dataset"
    )


if __name__ == "__main__":
    main()