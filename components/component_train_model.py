import logging
import optuna
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier


class HyperparameterOptimization:
    """
    Class for doing hyperparameter optimization for classification models.
    """

    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Initialize the class with the training and test data."""
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize_randomforest(self, trial: optuna.Trial) -> float:
        """
        Method for optimizing Random Forest for classification.
        """
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_lightgbm(self, trial: optuna.Trial) -> float:
        """
        Method for Optimizing LightGBM for classification.
        """
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.99)
        clf = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_xgboost(self, trial: optuna.Trial) -> float:
        """
        Method for Optimizing XGBoost for classification.
        """
        param = {
            "max_depth": trial.suggest_int("max_depth", 1, 30),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-7, 10.0),
            "n_estimators": trial.suggest_int("n_estimators", 1, 200),
        }
        clf = xgb.XGBClassifier(**param)
        clf.fit(self.x_train, self.y_train)
        val_accuracy = clf.score(self.x_test, self.y_test)
        return val_accuracy


class ModelTraining:
    """
    Class for training classification models.
    """

    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Initialize the class with the training and test data."""
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def random_forest_trainer(self, fine_tuning: bool = True):
        """
        It trains the random forest model for classification.
        """
        logging.info("Started training Random Forest model.")
        try:
            if fine_tuning:
                hyper_opt = HyperparameterOptimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_randomforest, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                max_depth = trial.params["max_depth"]
                min_samples_split = trial.params["min_samples_split"]
                print("Best parameters: ", trial.params)
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                )
                clf.fit(self.x_train, self.y_train)
                return clf
            else:
                model = RandomForestClassifier(
                    n_estimators=152, max_depth=20, min_samples_split=17
                )
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training Random Forest model")
            logging.error(e)
            return None

    def lightgbm_trainer(self, fine_tuning: bool = True):
        """
        It trains the LightGBM model for classification.
        """
        logging.info("Started training LightGBM model.")
        try:
            if fine_tuning:
                hyper_opt = HyperparameterOptimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_lightgbm, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                max_depth = trial.params["max_depth"]
                learning_rate = trial.params["learning_rate"]
                clf = LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                )
                clf.fit(self.x_train, self.y_train)
                return clf
            else:
                model = LGBMClassifier(
                    n_estimators=200, learning_rate=0.01, max_depth=20
                )
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training LightGBM model.")
            logging.error(e)
            return None

    def xgboost_trainer(self, fine_tuning: bool = True):
        """
        It trains the XGBoost model for classification.
        """
        logging.info("Started training XGBoost model.")
        try:
            if fine_tuning:
                hy_opt = HyperparameterOptimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hy_opt.optimize_xgboost, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                learning_rate = trial.params["learning_rate"]
                max_depth = trial.params["max_depth"]
                clf = xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                )
                clf.fit(self.x_train, self.y_train)
                return clf
            else:
                model = xgb.XGBClassifier(
                    n_estimators=200, learning_rate=0.01, max_depth=20
                )
                model.fit(self.x_train, self.y_train)
                return model

        except Exception as e:
            logging.error("Error in training XGBoost model.")
            logging.error(e)
            return None
