import pandas as pd    #load the data, statistics
import seaborn as sns   #visualize the data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
import matplotlib.pyplot as plt

# Choose Specifications
print("This script trains a model to predict the survival of passengers on the Titanic.")
print("What model would you like to use? (I would recommend a decision tree classifier)")
print("1. Decision tree classifier")
print("2. Logistic regression")

while True:
    model_choice = input("Input the number of your choice: ")
    if model_choice == '1':
        model = 'a decision tree classifier'
        break
    elif model_choice == '2':
        model = 'logistic regression'
        break
    else:
        print("Invalid choice. Please enter 1 or 2.")

print("What metric is most important to you?")
print("1. Accuracy")
print("2. Precision")
print("3. Recall")
print("4. All of the above")

while True:
    choice = input("Input the number of your choice: ")
    if choice == '1':
        print(f"This model will optimize for accuracy using {model}.")
        metric = 'all'
        break
    elif choice == '2':
        print(f"This model will optimize for precision using {model}.")
        metric = 'precision'
        break
    elif choice == '3':
        print(f"This model will optimize for recall using {model}.")
        metric = 'all'
        break
    elif choice == '4':
        print(f"This model will optimize for all metrics using {model}.")
        metric = 'all'
        break
    else:
        print("Invalid choice. Please enter a number between 1 and 4.")

#end choose specifications

#preprocess data
train_df = pd.read_csv('../input/train.csv')

selected_columns = train_df.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)

#1 is male, 2 is female
selected_columns['Sex'] = LabelEncoder().fit_transform(selected_columns['Sex'])
preprocessor = ColumnTransformer(
    transformers=[
        ('pclass', OneHotEncoder(drop='first'), ['Pclass'])
    ], remainder='passthrough')

# Specializing for precision
if (model == 'logistic regression' or metric == 'precision'):
    selected_columns['<10 yrs'] = train_df['Age'].apply(lambda x: 1 if x < 10 else 0)
    selected_columns['>60 yrs'] = train_df['Age'].apply(lambda x: 1 if x > 60 else 0)
    X = selected_columns[['Sex', 'Pclass', '<10 yrs', '>60 yrs']]
else: 
    X = selected_columns[['Sex', 'Pclass', 'Age']]

y = selected_columns['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)
#end preprocess data

#train model
if(model == 'a decision tree classifier'):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', DecisionTreeClassifier())])
else:
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression())])

pipeline.fit(X_train, y_train)
#end train model

#evaluate model
probabilities = pipeline.predict_proba(X_test)[:, 1]

#choose threshold
print("Do you want to choose a threshold for the model?")
choose_threshold = input("(y/n): ")

if(choose_threshold == 'n'):
    print("The model will use the default threshold of 0.5.")
    threshold = 0.5
else:
    print("Would you like to see precision-recall curve and precision-recall vs threshold curve to help you choose a threshold?")
    print("If yes, please close both graphs to process.")
    show_graph = input("(y/n): ")
    if show_graph == 'y':
        precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
        # Plot precision-recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.', label='Decision Tree')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        #end plot precision-recall curve

        #plot precision-recall vs threshold curve
        plt.figure(figsize=(10, 8))
        plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
        plt.plot(thresholds, recall[:-1], 'r-', label='Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision and Recall vs. Threshold')
        plt.legend()
        plt.grid(True)
        plt.show()
        #end plot precision-recall vs threshold curve
    threshold = float(input("Input the threshold: "))

predictions = (probabilities >= threshold).astype(int)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1: {f1:.2f}')
print("Confusion Matrix:")
print(cm)
    
#end evaluate model