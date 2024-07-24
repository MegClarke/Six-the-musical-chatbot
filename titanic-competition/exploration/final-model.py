import pandas as pd  # load the data, statistics
import seaborn as sns  # visualize the data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)
import matplotlib.pyplot as plt


def get_valid_input(prompt, valid_options):
    while True:
        user_input = input(prompt).lower()
        if user_input in valid_options:
            return user_input
        else:
            print(
                f"Invalid input. Please enter one of the following: {', '.join(valid_options)}"
            )


def get_valid_threshold(prompt):
    while True:
        try:
            threshold = float(input(prompt))
            if 0 <= threshold <= 1:
                return threshold
            else:
                print("Invalid input. Please enter a float between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a float between 0 and 1.")


# Choose Specifications
print(
    "This script trains a model to predict the survival of passengers on the Titanic.\n"
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
        print("To optimize precision, I recommend using a Decision Tree Classifier.")
        metric = "precision"
        break
    elif choice == "3":
        print("To optimize recall, I recommend using a Logistic Regression model.")
        metric = "all"
        break
    elif choice == "4":
        print("To optimize all metrics,I recommend using a Decision Tree Classifier.")
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
        model = "dt"
        break
    elif model_choice == "2":
        model = "lr"
        break
    else:
        print("Invalid choice. Please enter 1 or 2.")


# end choose specifications

# preprocess data
train_df = pd.read_csv("../input/train.csv")

selected_columns = train_df.drop(
    ["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1
)

# 1 is male, 2 is female
selected_columns["Sex"] = LabelEncoder().fit_transform(selected_columns["Sex"])
preprocessor = ColumnTransformer(
    transformers=[("pclass", OneHotEncoder(drop="first"), ["Pclass"])],
    remainder="passthrough",
)

# Specializing for precision
if model == "lr" or metric == "precision":
    selected_columns["<10 yrs"] = train_df["Age"].apply(lambda x: 1 if x < 10 else 0)
    selected_columns[">60 yrs"] = train_df["Age"].apply(lambda x: 1 if x > 60 else 0)
    X = selected_columns[["Sex", "Pclass", "<10 yrs", ">60 yrs"]]
else:
    X = selected_columns[["Sex", "Pclass", "Age"]]

y = selected_columns["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3, stratify=y
)
# end preprocess data

# train model
if model == "dt":
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", DecisionTreeClassifier())]
    )
else:
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
    )

pipeline.fit(X_train, y_train)
# end train model

# evaluate model
probabilities = pipeline.predict_proba(X_test)[:, 1]

# choose threshold
print("\nDo you want to choose a threshold for the model?")
choose_threshold = get_valid_input("(y/n): ", ["y", "n"])

if choose_threshold == "n":
    print("\nThe model will use the default threshold of 0.5.")
    threshold = 0.5
else:
    print(
        "Would you like to see precision-recall curve to help you choose a threshold?"
    )
    show_graph = get_valid_input("(y/n): ", ["y", "n"])

    if show_graph == "y":
        precision, recall, thresholds = precision_recall_curve(y_test, probabilities)

        # Plot precision-recall curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # Plot precision-recall curve
        ax1.plot(recall, precision, marker='.', label='Decision Tree')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision-Recall Curve')
        ax1.legend()

        # Plot precision-recall vs threshold curve
        ax2.plot(thresholds, precision[:-1], 'b-', label='Precision')
        ax2.plot(thresholds, recall[:-1], 'r-', label='Recall')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision and Recall vs. Threshold')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    threshold = get_valid_threshold("Input the threshold: ")

print(f"\nThreshold: {threshold}")

predictions = (probabilities >= threshold).astype(int)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1: {f1:.2f}")
print("Confusion Matrix:")
print(cm)

# end evaluate model
