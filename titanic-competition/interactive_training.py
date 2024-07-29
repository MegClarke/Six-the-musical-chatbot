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
from typing import Tuple


def choose_model_and_metric() -> Tuple[str, str]:
    """
    Prompts the user to choose a model and metric for training.

    Returns:
        Tuple[str, str]: The chosen model and metric.
    """
    print(
        "\nThis script trains a model to predict the survival of passengers on the Titanic.\n"
    )

    print("What metric is most important to you?")
    print("1. Accuracy")
    print("2. Precision")
    print("3. Recall")
    print("4. All of the above")

    while True:
        choice = input("\nInput the number of your choice: ")
        if choice == "1":
            print("To optimize accuracy, I recommend using a Decision Tree Classifier.")
            metric = "all"
            break
        elif choice == "2":
            print(
                "To optimize precision, I recommend using a Decision Tree Classifier."
            )
            metric = "precision"
            break
        elif choice == "3":
            print("To optimize recall, I recommend using a Logistic Regression model.")
            metric = "all"
            break
        elif choice == "4":
            print(
                "To optimize all metrics, I recommend using a Decision Tree Classifier."
            )
            metric = "all"
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

    print("\nWhat model would you like to use?")
    print("1. Decision tree classifier")
    print("2. Logistic regression")

    while True:
        model_choice = input("\nInput the number of your choice: ")
        if model_choice == "1":
            model = "decision"
            break
        elif model_choice == "2":
            model = "logistic"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    return model, metric


def get_valid_input(prompt: str, valid_options: list) -> str:
    """
    Prompts the user to enter a valid input from the given options.

    Args:
        prompt (str): The input prompt to display to the user.
        valid_options (list): A list of valid options.

    Returns:
        str: The valid user input.
    """
    while True:
        user_input = input(prompt).lower()
        if user_input in valid_options:
            return user_input
        else:
            print(
                f"Invalid input. Please enter one of the following: {', '.join(valid_options)}"
            )


def get_valid_threshold(prompt: str) -> float:
    """
    Prompts the user to enter a valid threshold between 0 and 1.

    Args:
        prompt (str): The input prompt to display to the user.

    Returns:
        float: The valid threshold entered by the user.
    """
    while True:
        try:
            threshold = float(input(prompt))
            if 0 <= threshold <= 1:
                return threshold
            else:
                print("Invalid input. Please enter a float between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a float between 0 and 1.")


def preprocess_data(model: str, metric: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, object]:
    """
    Preprocesses the Titanic dataset for training.

    Args:
        model (str): The chosen model type.
        metric (str): The chosen metric type.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, object]: The preprocessed training and test data, and the preprocessor object.
    """
    train_df = pd.read_csv("input/train.csv")
    selected_columns = train_df.drop(
        ["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1
    )
    selected_columns["Sex"] = LabelEncoder().fit_transform(selected_columns["Sex"])
    preprocessor = ColumnTransformer(
        transformers=[("pclass", OneHotEncoder(drop="first"), ["Pclass"])],
        remainder="passthrough",
    )

    if model == "logistic" or metric == "precision":
        selected_columns["<10 yrs"] = train_df["Age"].apply(
            lambda x: 1 if x < 10 else 0
        )
        selected_columns[">60 yrs"] = train_df["Age"].apply(
            lambda x: 1 if x > 60 else 0
        )
        X = selected_columns[["Sex", "Pclass", "<10 yrs", ">60 yrs"]]
    else:
        X = selected_columns[["Sex", "Pclass", "Age"]]

    y = selected_columns["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=3, stratify=y
    )
    return X_train, X_test, y_train, y_test, preprocessor


def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer, model: str) -> Pipeline:
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
    if model == "decision":
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", DecisionTreeClassifier()),
            ]
        )
    else:
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
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


def choose_threshold(y_test: pd.Series, probabilities: pd.Series) -> float:
    """
    Prompts the user to choose a threshold for the model and optionally plots graphs to help in the decision.

    Args:
        y_test (pd.Series): The true labels for the test data.
        probabilities (pd.Series): The predicted probabilities for the test data.

    Returns:
        float: The chosen threshold.
    """
    print("\nDo you want to choose a threshold for the model?")
    choose_threshold = get_valid_input("(y/n): ", ["y", "n"])

    if choose_threshold == "n":
        print("\nThe model will use the default threshold of 0.5.")
        return 0.5
    else:
        print(
            "Would you like to see precision-recall curve to help you choose a threshold?"
        )
        show_graph = get_valid_input("(y/n): ", ["y", "n"])

        if show_graph == "y":
            plot_graphs(y_test, probabilities)

        return get_valid_threshold("Input the threshold: ")


def evaluate_model(y_test: pd.Series, probabilities: pd.Series, threshold: float) -> None:
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
    Main function to execute the model training and evaluation pipeline.

    This function guides the user through choosing a model and metric, 
    preprocesses the data, trains the selected model, predicts probabilities, 
    allows the user to choose a threshold, and evaluates the model based on 
    the chosen threshold.
    """
    model, metric = choose_model_and_metric()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(model, metric)
    pipeline = train_model(X_train, y_train, preprocessor, model)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    threshold = choose_threshold(y_test, probabilities)
    evaluate_model(y_test, probabilities, threshold)


if __name__ == "__main__":
    main()
