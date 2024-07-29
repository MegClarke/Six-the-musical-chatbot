from unittest import mock, TestCase
import pytest

import pandas as pd

from ...final_model import validate_csv, validate_model, validate_threshold, preprocess_data
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


mock_csv_data = """
Survived,Sex,Pclass,Age
0,male,1,22
1,male,1,7
0,female,3,38
1,female,2,26
"""

mock_df = pd.DataFrame({
    'Id': [1, 2, 3, 4, 5],
    'Survived': [1, 0, 1, 1, 0],
    'Sex': ['male', 'female', 'male', 'female', 'female'],
    'Pclass': [1, 2, 3, 1, 2],
    'Age': [22, 38, 6, 35, 61]
})


class TestArgParse:
    # Test validate_csv
    def test_validate_csv_valid_file(self, tmpdir):
        # Create a temporary CSV file
        file_path = tmpdir.join("valid.csv")
        file_path.write(mock_csv_data)
        
        # This should not raise any exceptions
        validate_csv(str(file_path))

    def test_validate_csv_missing_file(self):
        with pytest.raises(FileNotFoundError):
            validate_csv("non_existent_file.csv")

    def test_validate_csv_missing_columns(self, tmpdir):
        # Create a temporary CSV file with missing columns
        invalid_csv_data = "Survived,Sex\n1,male\n0,female"
        file_path = tmpdir.join("invalid.csv")
        file_path.write(invalid_csv_data)
        
        with pytest.raises(ValueError):
            validate_csv(str(file_path))

    # Test validate_model
    def test_validate_model(self):
        validate_model("logistic_regression")
        validate_model("decision_trees")

        with pytest.raises(ValueError):
            validate_model("invalid_model")

    # Test validate_threshold
    def test_validate_threshold_valid(self):
        validate_threshold(0.5)
        validate_threshold(0.0)
        validate_threshold(1.0)

        with pytest.raises(ValueError):
            validate_threshold(1.5)
        with pytest.raises(ValueError):
            validate_threshold(-0.5)
        with pytest.raises(ValueError):
            validate_threshold("not_a_float")


# Test preprocess_data
class TestPreprocessData(TestCase):
    def setUp(self, tmpdir):
        # Patch the read_csv method
        self.patcher = mock.patch('pandas.read_csv', return_value=mock_df)
        self.mock_read_csv = self.patcher.start()

        file_path = tmpdir.join("mock.csv")
        file_path.write(mock_csv_data)
        
    def tearDown(self):
        # Stop the patcher
        self.patcher.stop()
    
    def test_preprocess_data_decision_tree(self):
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data("mock.csv", "decision_trees")   
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert isinstance(preprocessor, ColumnTransformer)

        self.assertEqual(X_train.shape, (4, 4))
        self.assertEqual(X_test.shape, (1, 4))
        self.assertEqual(y_train.shape, (4,))
        self.assertEqual(y_test.shape, (1,))

        expected_columns = ['Sex', 'Pclass', '<10 yrs', '>60 yrs']
        self.assertTrue(all(col in X_train.columns for col in expected_columns))

    def test_preprocess_data_logistic_regression(self):
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data("mock.csv", "logistic_regression")
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert isinstance(preprocessor, ColumnTransformer)

        self.assertEqual(X_train.shape, (4, 3))
        self.assertEqual(X_test.shape, (1, 3))
        self.assertEqual(y_train.shape, (4,))
        self.assertEqual(y_test.shape, (1,))

        expected_columns = ['Sex', 'Pclass', 'Age']
        self.assertTrue(all(col in X_train.columns for col in expected_columns))


'''
class TestTrainModel:
    def test_train_model(self):
        X_train = pd.DataFrame({
            'Sex': ['male', 'female', 'male', 'female', 'female'],
            'Pclass': [1, 2, 3, 1, 2],
            'Age': [22, 38, 6, 35, 61]
        })

        y_train = pd.Series([1, 0, 1, 1, 0])

        model = train_model(X_train, y_train, preprocessor, "decision_trees")
        assert isinstance(model, DecisionTreeClassifier)
        
        model = train_model(X_train, y_train, preprocessor, "logistic_regression")
        assert isinstance(model, LogisticRegression)
'''
