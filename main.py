#!/usr/bin/env python3

from lightning.pytorch.cli import LightningCLI

from demo.data import DemoDataModule

from demo.demo_simplevit_lit import DemoSimpleVitLit
from demo.demo_vit_lit import DemoVitLit

import logging

logger = logging.getLogger(__name__)

class ViT(DemoVitLit):
    def configure_optimizers(self):
        logger.info(f"⚡ Using Vit ⚡")
        return super().configure_optimizers()

class SimpleViT(DemoSimpleVitLit):
    def configure_optimizers(self):
        logger.info(f"⚡ Using SimpleViT ⚡")
        return super().configure_optimizers()

def main():
    level = logging.INFO
    logging.basicConfig(level=level)

    cli = LightningCLI(
        datamodule_class=DemoDataModule,
    )

if __name__ == "__main__":
    main()
