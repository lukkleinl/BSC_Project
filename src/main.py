from dataset.dataset import Dataset
from preprocessing.config import Config


def main():
    config = Config()
    dataset = Dataset(config)
    dataset.transform_data()
    dataset.train_test_split()
    dataset.train()


if __name__ == "__main__":
    main()
