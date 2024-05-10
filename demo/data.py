from typing import Optional
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import logging
import lightning as L

logger = logging.getLogger(__name__)

class DemoDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "mnist", 
        download: bool = True,
        batch_size: int = 32,
        num_workers=1,
        pin_memory=False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.download = download

    def setup(self, stage: Optional[str] = None):
        transform = transforms.Compose([transforms.ToTensor()])

        self.mnist_test = MNIST(self.data_dir, download=self.download, train=False, transform=transform)
        self.mnist_predict = MNIST(self.data_dir, download=self.download, train=False, transform=transform)
        mnist_full = MNIST(self.data_dir, download=self.download, train=True, transform=transform)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        if self.mnist_train is None:
            raise ValueError("Setup needs to be called first")

        return DataLoader(
            dataset=self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        if self.mnist_val is None:
            raise ValueError("Setup needs to be called first")

        return DataLoader(
            dataset=self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self):
        if self.mnist_test is None:
            raise ValueError("Setup needs to be called first")
        
        return DataLoader(
            dataset=self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        if self.mnist_predict is None:
            raise ValueError("Setup needs to be called first")

        return DataLoader(
            dataset=self.mnist_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )


