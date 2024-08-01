import os
import tempfile
from unittest import TestCase, mock

import pandas as pd
import pytest
from final_model import (
    DecisionTreeClassifier,
    LogisticRegression,
    Pipeline,
    preprocess_data,
    train_model,
    validate_csv,
    validate_model,
    validate_threshold,
)
from sklearn.compose import ColumnTransformer

mock_csv_data = """
Survived,Sex,Pclass,Age
0,male,1,22
1,male,1,7
0,female,3,38
1,female,2,26
"""

mock_df = pd.DataFrame(
    {
        "Id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Survived": [1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
        "Sex": ["male", "female", "male", "female", "female", "male", "female", "male", "male", "female"],
        "Pclass": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        "Age": [22, 38, 6, 35, 61, 50, 30, 40, 20, 45],
    }
)


class TestArgParse:
    """
    Tests for argument parsing functions, including CSV, model, and threshold validation.
    """
     
    def test_validate_csv_valid_file(self, tmpdir):
        """
        Condition:
        A valid CSV file with the correct columns and data format is provided.
        
        Expected:
        No exceptions should be raised.
        """
        file_path = tmpdir.join("valid.csv")
        file_path.write(mock_csv_data)

        validate_csv(str(file_path))

    def test_validate_csv_missing_file(self):
        """
        Condition:
        A non-existent CSV file is provided.
        
        Expected:
        FileNotFoundError should be raised.
        """
        with pytest.raises(FileNotFoundError):
            validate_csv("non_existent_file.csv")

    def test_validate_csv_missing_columns(self, tmpdir):
        """
        Condition:
        A CSV file with missing columns is provided.
        
        Expected:
        ValueError should be raised.
        """
        invalid_csv_data = "Survived,Sex\n1,male\n0,female"
        file_path = tmpdir.join("invalid.csv")
        file_path.write(invalid_csv_data)

        with pytest.raises(ValueError):
            validate_csv(str(file_path))

    def test_validate_model_valid(self):
        """
        Condition:
        Valid model names ("logistic_regression", "decision_trees") are provided.
        
        Expected:
        No exceptions should be raised.
        """
        validate_model("logistic_regression")
        validate_model("decision_trees")

    def test_validate_model_invalid(self):
        """
        Condition:
        Invalid model name is provided.
        
        Expected:
        ValueError should be raised.
        """
        with pytest.raises(ValueError):
            validate_model("invalid_model")

    def test_validate_threshold_valid(self):
        """
        Condition:
        Valid threshold values (0.0, 0.5, 1.0) are provided.
        
        Expected:
        No exceptions should be raised.
        """
        validate_threshold(0.5)
        validate_threshold(0.0)
        validate_threshold(1.0)

    def test_validate_threshold_invalid(self):
        """
        Condition:
        Invalid threshold values (e.g., -0.5, 1.5, "not_a_float") are provided.
        
        Expected:
        ValueError should be raised.
        """
        with pytest.raises(ValueError):
            validate_threshold(1.5)
        with pytest.raises(ValueError):
            validate_threshold(-0.5)
        with pytest.raises(ValueError):
            validate_threshold("not_a_float")


class TestPreprocessData(TestCase):
    """
    Tests for the preprocess_data function, ensuring it processes data correctly for different models.
    """
    def setUp(self):
        patcher1 = mock.patch("pandas.read_csv", return_value=mock_df)
        patcher2 = mock.patch(
            "sklearn.compose.ColumnTransformer",
            mock.MagicMock(name="ColumnTransformer"),
        )
        
        self.mock_read_csv = patcher1.start()
        self.mock_column_transformer = patcher2.start()
        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)

    def test_preprocess_data_decision_tree(self):
        """
        Condition:
        The preprocess_data function is called with 'decision_trees' model type.
        
        Expected:
        The data should be split into training sets containing columns ["Sex", "Pclass", "Age"], and test sets containing the column ["Survived"].
        """
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
            "file.csv", "decision_trees"
        )

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert isinstance(preprocessor, ColumnTransformer)

        self.assertEqual(X_train.shape, (8, 3))
        self.assertEqual(X_test.shape, (2, 3))
        self.assertEqual(y_train.shape, (8,))
        self.assertEqual(y_test.shape, (2,))

        expected_columns = ["Sex", "Pclass", "Age"]
        self.assertTrue(all(col in X_train.columns for col in expected_columns))

    def test_preprocess_data_logistic_regression(self):
        """
        Condition:
        The preprocess_data function is called with 'logistic_regression' model type.
        
        Expected:
        The data should be split into training sets containing columns ["Sex", "Pclass", "<10 yrs", ">60 yrs"], and test sets containing the column ["Survived"].
        """
        X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
            "file.csv", "logistic_regression"
        )

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert isinstance(preprocessor, ColumnTransformer)

        self.assertEqual(X_train.shape, (8, 4))
        self.assertEqual(X_test.shape, (2, 4))
        self.assertEqual(y_train.shape, (8,))
        self.assertEqual(y_test.shape, (2,))

        expected_columns = ["Sex", "Pclass", "<10 yrs", ">60 yrs"]
        self.assertTrue(all(col in X_train.columns for col in expected_columns))


class TestTrainModel(TestCase):
    """
    Tests for the train_model function, ensuring models are trained correctly.
    """
    def setUp(self):
        patcher1 = mock.patch("final_model.Pipeline", autospec=True)
        patcher2 = mock.patch("sklearn.compose.ColumnTransformer", autospec=True)

        self.mock_pipeline_class = patcher1.start()
        self.mock_column_transformer = patcher2.start()

        self.mock_pipeline_instance = self.mock_pipeline_class.return_value
        self.mock_pipeline_instance.fit = mock.Mock()

        self.X_train = pd.DataFrame(
            {
                "Sex": ["male", "female", "male", "female", "female"],
                "Pclass": [1, 2, 3, 1, 2],
                "Age": [22, 38, 6, 35, 61],
            }
        )

        self.y_train = pd.Series([1, 0, 1, 1, 0])

        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)

    def test_train_model_decision_trees(self):
        """
        Condition:
        The train_model function is called with 'decision_trees' model type.
        
        Expected:
        The classifier in the pipeline should be DecisionTreeClassifier and the model should be trained using the provided training data.
        """
        train_model(
            self.X_train, self.y_train, self.mock_column_transformer, "decision_trees"
        )
        self.mock_pipeline_class.assert_called_once()
        _, kwargs = self.mock_pipeline_class.call_args
        steps = kwargs['steps']
        self.assertEqual(steps[1][0], 'classifier')
        self.assertIsInstance(steps[1][1], DecisionTreeClassifier)

        self.mock_pipeline_instance.fit.assert_called_once_with(self.X_train, self.y_train)

    def test_train_model_logistic_regression(self):
        """
        Condition:
        The train_model function is called with 'logistic_regression' model type.
        
        Expected:
        The model should be trained using the provided training data and column transformer.
        """
        train_model(
            self.X_train,
            self.y_train,
            self.mock_column_transformer,
            "logistic_regression",
        )
        self.mock_pipeline_class.assert_called_once()
        _, kwargs = self.mock_pipeline_class.call_args
        steps = kwargs['steps']
        self.assertEqual(steps[1][0], 'classifier')
        self.assertIsInstance(steps[1][1], LogisticRegression)

        self.mock_pipeline_instance.fit.assert_called_once_with(self.X_train, self.y_train)
