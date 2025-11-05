#!/usr/bin/env python3

from lightning.pytorch.cli import LightningCLI

from demo.data import DemoDataModule

import logging

logger = logging.getLogger(__name__)

def main():
    level = logging.INFO
    logging.basicConfig(level=level)

    cli = LightningCLI()

if __name__ == "__main__":
    main()
