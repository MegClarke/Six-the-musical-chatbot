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
from final_model import preprocess_data, train_model, evaluate_model, plot_graphs


def choose_model_and_metric() -> tuple[str, str]:
    """
    Prompts the user to choose a model and metric for training.

    Returns:
        tuple[str, str]: The chosen model and metric.
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
            model = ModelType.DEC_TREES.value
            break
        elif model_choice == "2":
            model = ModelType.LOG_REG.value
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
        print(f"\nThe model will use the default threshold of {DEFAULT_THRESHOLD}.")
        return DEFAULT_THRESHOLD
    else:
        print(
            "Would you like to see precision-recall curve to help you choose a threshold?"
        )
        show_graph = get_valid_input("(y/n): ", ["y", "n"])

        if show_graph == "y":
            plot_graphs(y_test, probabilities)

        return get_valid_threshold("Input the threshold: ")


def main() -> None:
    """
    Main function to execute the model training and evaluation pipeline.

    This function guides the user through choosing a model and metric, 
    preprocesses the data, trains the selected model, predicts probabilities, 
    allows the user to choose a threshold, and evaluates the model based on 
    the chosen threshold.
    """
    model, metric = choose_model_and_metric()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data('input/train.csv', model)
    pipeline = train_model(X_train, y_train, preprocessor, model)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    threshold = choose_threshold(y_test, probabilities)
    evaluate_model(y_test, probabilities, threshold)


if __name__ == "__main__":
    main()
