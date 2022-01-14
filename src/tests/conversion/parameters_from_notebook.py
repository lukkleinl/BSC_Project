def get_params_notebook():
    Paths = {
        "raw_data_path": "data/raw/",
        "preprocessing_path": "data/preprocessed/",
        "processed_path": "data/processed/",
        "model_path": "data/model/",
        "notebook_path": "data/notebooks/Wine_Quality"
    }

    Converter = {
        "convert_preprocessing_steps": True,
        "parameter_conversion": True,
        "notebook_name": "prediction-of-quality-of-wine.ipynb",
        "preprocessing_tags":
            [
                "transform_data_func"
            ]
    }

    Loader = {
        "name": "URLLoader",
        "name_of_file": "raw_data.csv",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    }

    Preprocessing_Steps = {
        "clean_data": {
            "conversion": False,
            "params": {
                "target": "quality",
                "drop_columns":
                    [
                        "alcohol"
                        "sul"
                    ]
            }
        },
        "transform_data": {
            "conversion": True,
            "params": {
                "target": "quality"
            }
        }
    }

    Dataset = {
        "parameter_train_test_split":
            [
                "0.2",
                "42"
            ]
    }

    Model = {
        "file_name": "rf_model_new_nb.pkl",
        "ensemble_model": "RandomForestClassifier",
        "model_config": {"n_estimators": 100, "random_state": 42},
        "export_result": True
    }

    params = dict()
    params.update({"Paths": Paths, "Converter": Converter, "Loader": Loader, "Preprocessing_Steps": Preprocessing_Steps,
                   "Dataset": Dataset, "Model": Model})

    return params