class Config:
    # Data prep
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    raw_data_file = "data/winequality-red.csv"
    test_size = 0.2
    random_state = 42

    # train
    processed_train = "/home/lukas/PycharmProjects/BSC_Project/src/data/train.csv"
    ensemble_model = "RandomForestClassifier"
    model_n_estimators = 100
    model_random_state = 42
    model_path = "/home/lukas/PycharmProjects/BSC_Project/src/data/model/rf_model.pkl"

    # predict
    processed_test = "/home/lukas/PycharmProjects/BSC_Project/src/data/test.csv"
    predicted_file = "/home/lukas/PycharmProjects/BSC_Project/src/data/predict.csv"
    export_result = True
