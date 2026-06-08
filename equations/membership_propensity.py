

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def logistic_membership_propensity(data, covariates, source, prediction):

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(data[covariates], data[source])
    out = data.copy()
    out[prediction] = model.predict_proba(data[covariates])[:, 1]
    return out


def xgboost_membership_propensity(data, covariates, source, prediction):

    from xgboost import XGBClassifier  # heavy + optional, imported lazily
    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                          eval_metric="logloss", random_state=0)
    model.fit(data[covariates], data[source])
    out = data.copy()
    out[prediction] = model.predict_proba(data[covariates])[:, 1]
    return out


def neural_network_membership_propensity(data, covariates, source, prediction):
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    keras.utils.set_random_seed(0)

    scaler = StandardScaler().fit(data[covariates])
    X = scaler.transform(data[covariates])
    y = data[source].to_numpy()

    # hold out a slice for post-hoc calibration (the net never sees y_cal)
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y)

    reg = keras.regularizers.l2(1e-3)
    net = keras.Sequential([
        keras.Input(shape=(len(covariates),)),
        keras.layers.Dense(16, activation="relu", kernel_regularizer=reg),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu", kernel_regularizer=reg),
        keras.layers.Dense(1),  # logits (no activation)
    ])
    net.compile(optimizer="adam",
                loss=keras.losses.BinaryCrossentropy(from_logits=True))
    net.fit(X_fit, y_fit, validation_split=0.2, epochs=300, batch_size=32, verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True)])

    # Platt / temperature scaling: logistic on the net's held-out logits
    platt = LogisticRegression().fit(net.predict(X_cal, verbose=0), y_cal)

    out = data.copy()
    out[prediction] = platt.predict_proba(net.predict(X, verbose=0))[:, 1]
    return out