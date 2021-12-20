

class Config:
    # Data prep
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    raw_data_file = "data/wine_quality/raw_data.csv"
    test_size = 0.2
    random_state = 42
    target = "quality"

    # train
    processed_train = "data/wine_quality/train.csv"
    ensemble_model = "RandomForestClassifier"
    model_n_estimators = 100
    model_random_state = 42
    model_path = "data/model/rf_model.pkl"

    # predict
    processed_test = "data/wine_quality/test.csv"
    predicted_file = "data/wine_quality/predict.csv"
    export_result = True
