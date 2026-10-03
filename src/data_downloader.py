from torchvision import datasets, transforms

class DataDownloader:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir

    def download_mnist(self):
        transform = transforms.ToTensor()

        train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform
        )

        return train_dataset, test_dataset