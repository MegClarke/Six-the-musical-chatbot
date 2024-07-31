import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from constants import SURVIVED, SEX, PCLASS, AGE, UNDER_10_YEARS, OVER_60_YEARS, DEFAULT_THRESHOLD, TEST_SIZE, RANDOM_STATE, ModelType


def validate_csv(file_path: str) -> None:
    """
    Validate that the CSV file exists and contains the required columns.

    Args:
        file_path (str): Path to the CSV file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is missing required columns.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    df = pd.read_csv(file_path)

    required_columns = {SURVIVED, SEX, PCLASS, AGE}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"The file {file_path} is missing the following required columns: {', '.join(missing_columns)}"
        )


def validate_model(model: str) -> None:
    """
    Validate the model type.

    Args:
        model (str): The model type to validate.

    Raises:
        ValueError: If the model type is invalid.
    """
    valid_models = { ModelType.LOG_REG.value , ModelType.DEC_TREES.value}
    if model not in valid_models:
        raise ValueError(
            f"Invalid model type. Expected one of {valid_models}, got '{model}'."
        )


def validate_threshold(threshold: float) -> None:
    """
    Validate the threshold value.

    Args:
        threshold (float): The threshold value to validate.

    Raises:
        ValueError: If the threshold is not a float or not between 0 and 1.
    """
    try:
        threshold = float(threshold)
    except ValueError:
        raise ValueError(f"Threshold should be a float. Got '{threshold}'.")

    if not (0 <= threshold <= 1):
        raise ValueError(f"Threshold should be between 0 and 1. Got '{threshold}'.")


def parse_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a classifier model on the Titanic dataset."
    )

    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the Titanic CSV file. Must contain columns: Survived, Sex, Pclass, Age.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=ModelType.DEC_TREES.value,
        choices=[ModelType.LOG_REG.value, ModelType.DEC_TREES.value],
        help="Model to use: logistic_regression or decision_trees. Default is decision_trees.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Threshold value between 0 and 1. Default is 0.5.",
    )

    args = parser.parse_args()

    validate_csv(args.csv_file)
    validate_model(args.model)
    validate_threshold(args.threshold)

    return args


def preprocess_data(
    file_path: str, model: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Preprocesses data for training and testing.

    Args:
        file_path (str): Path to the CSV file.
        model (str): The model type to decide preprocessing steps.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
            - X_train: Training features
            - X_test: Test features
            - y_train: Training labels
            - y_test: Test labels
            - preprocessor: The preprocessing transformer.
    """

    train_df = pd.read_csv(file_path)
    selected_columns = train_df
    selected_columns[SEX] = LabelEncoder().fit_transform(selected_columns[SEX])
    preprocessor = ColumnTransformer(
        transformers=[(PCLASS, OneHotEncoder(drop="first"), [PCLASS])],
        remainder="passthrough",
    )

    if model == ModelType.LOG_REG.value:
        selected_columns[UNDER_10_YEARS] = train_df[AGE].apply(
            lambda x: 1 if x < 10 else 0
        )
        selected_columns[OVER_60_YEARS] = train_df[AGE].apply(
            lambda x: 1 if x > 60 else 0
        )
        X = selected_columns[[SEX, PCLASS, UNDER_10_YEARS, OVER_60_YEARS]]
    else:
        X = selected_columns[[SEX, PCLASS, AGE]]

    y = selected_columns[SURVIVED]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, preprocessor


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    model: str,
) -> Pipeline:
    """
    Trains a machine learning model using the given data.

    Args:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels.
        preprocessor (ColumnTransformer): The preprocessor object.
        model (str): The chosen model type.

    Returns:
        Pipeline: The trained model pipeline.
    """
    if model == ModelType.DEC_TREES.value:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", DecisionTreeClassifier()),
            ]
        )
    else:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression()),
            ]
        )
    pipeline.fit(X_train, y_train)
    return pipeline


def plot_graphs(y_test: pd.Series, probabilities: pd.Series) -> None:
    """
    Plots precision-recall curves and precision-recall vs threshold curves.

    Args:
        y_test (pd.Series): The true labels for the test data.
        probabilities (pd.Series): The predicted probabilities for the test data.
    """
    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot precision-recall curve
    ax1.plot(recall, precision, marker=".", label="Decision Tree")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall Curve")
    ax1.legend()

    # Plot precision-recall vs threshold curve
    ax2.plot(thresholds, precision[:-1], "b-", label="Precision")
    ax2.plot(thresholds, recall[:-1], "r-", label="Recall")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("Precision and Recall vs. Threshold")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_model(
    y_test: pd.Series, probabilities: pd.Series, threshold: float
) -> None:
    """
    Evaluates the model using various metrics and prints the results.

    Args:
        y_test (pd.Series): The true labels for the test data.
        probabilities (pd.Series): The predicted probabilities for the test data.
        threshold (float): The threshold for converting probabilities to class labels.
    """
    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print(f"\nThreshold: {threshold}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1: {f1:.2f}")
    print("Confusion Matrix:")
    print(cm)


def main() -> None:
    """
    Main entry point of the script. This function handles the entire workflow:

    1. Parses and validates command-line arguments.
    2. Preprocesses the Titanic dataset based on the specified model.
    3. Trains the selected machine learning model.
    4. Generates probability predictions on the test set.
    5. Plots precision-recall curves and precision-recall vs threshold curves.
    6. Evaluates the model using various metrics and prints the results.

    This function ensures that all necessary steps are executed in sequence,
    handling any exceptions that arise during processing.
    """
    try:
        # Parse and validate arguments
        args = parse_args()

        # Print or use the arguments for your application
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
            args.csv_file, args.model
        )
        pipeline = train_model(X_train, y_train, preprocessor, args.model)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        evaluate_model(y_test, probabilities, args.threshold)
        plot_graphs(y_test, probabilities)

        # Proceed with the rest of your script logic
        # e.g., load data, train model, etc.

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
