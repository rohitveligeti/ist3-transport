from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def _fit_outcome(data, covariates, outcome, source, treatment, value, prediction, model):

    arm = data[(data[source] == 1) & (data[treatment] == value)]
    model.fit(arm[covariates], arm[outcome])
    out = data.copy()
    out[prediction] = model.predict_proba(data[covariates])[:, 1]
    return out


# ----- logistic regression -----

def logistic_treatment_outcome(data, covariates, outcome, source, treatment, prediction):
    return _fit_outcome(data, covariates, outcome, source, treatment, 1, prediction,
                        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))


def logistic_control_outcome(data, covariates, outcome, source, treatment, prediction):
    return _fit_outcome(data, covariates, outcome, source, treatment, 0, prediction,
                        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))


# ----- random forest -----

def random_forest_treatment_outcome(data, covariates, outcome, source, treatment, prediction):
    return _fit_outcome(data, covariates, outcome, source, treatment, 1, prediction,
                        RandomForestClassifier(n_estimators=300, random_state=0))


def random_forest_control_outcome(data, covariates, outcome, source, treatment, prediction):
    return _fit_outcome(data, covariates, outcome, source, treatment, 0, prediction,
                        RandomForestClassifier(n_estimators=300, random_state=0))


# ----- xgboost -----

def xgboost_treatment_outcome(data, covariates, outcome, source, treatment, prediction):
    from xgboost import XGBClassifier  # heavy + optional, imported lazily
    return _fit_outcome(data, covariates, outcome, source, treatment, 1, prediction,
                        XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                      eval_metric="logloss", random_state=0))


def xgboost_control_outcome(data, covariates, outcome, source, treatment, prediction):
    from xgboost import XGBClassifier
    return _fit_outcome(data, covariates, outcome, source, treatment, 0, prediction,
                        XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                      eval_metric="logloss", random_state=0))


# ----- neural network (keras) -----

def _nn_outcome(data, covariates, outcome, source, treatment, value, prediction):
    """Fit a small feed-forward net on the source==1 & treatment==value arm;
    inputs standardized, sigmoid output -> P(outcome=1) for all rows."""
    from tensorflow import keras  # heavy + optional, imported lazily
    arm = data[(data[source] == 1) & (data[treatment] == value)]
    scaler = StandardScaler().fit(arm[covariates])
    net = keras.Sequential([
        keras.Input(shape=(len(covariates),)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    net.compile(optimizer="adam", loss="binary_crossentropy")
    net.fit(scaler.transform(arm[covariates]), arm[outcome].to_numpy(),
            epochs=50, batch_size=32, verbose=0)
    out = data.copy()
    out[prediction] = net.predict(scaler.transform(data[covariates]), verbose=0).ravel()
    return out


def neural_network_treatment_outcome(data, covariates, outcome, source, treatment, prediction):
    return _nn_outcome(data, covariates, outcome, source, treatment, 1, prediction)


def neural_network_control_outcome(data, covariates, outcome, source, treatment, prediction):
    return _nn_outcome(data, covariates, outcome, source, treatment, 0, prediction)