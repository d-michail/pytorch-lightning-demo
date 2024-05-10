#!/usr/bin/env python3

from lightning.pytorch.cli import LightningCLI

from demo.data import DemoDataModule
from demo.models import DemoLit
import logging


def main():
    level = logging.INFO
    logging.basicConfig(level=level)

    cli = LightningCLI(
        model_class=DemoLit,
        datamodule_class=DemoDataModule,
    )

if __name__ == "__main__":
    main()
