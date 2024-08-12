from enum import Enum

SURVIVED = "Survived"
SEX = "Sex"
PCLASS = "Pclass"
AGE = "Age"
UNDER_10_YEARS = "<10 yrs"
OVER_60_YEARS = ">60 yrs"

DEFAULT_THRESHOLD = 0.5
TEST_SIZE = 0.2
RANDOM_STATE = 3


class ModelType(Enum):
    LOG_REG = "logistic_regression"
    DEC_TREES = "decision_trees"
