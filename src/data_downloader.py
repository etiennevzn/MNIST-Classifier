from torchvision import datasets

class DataDownloader:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir

    def download_mnist(self):
        train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True
        )

        test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True
        )

        return train_dataset, test_dataset