from lightning.pytorch.callbacks import Callback
import pytorch_lightning as pl


class OverrideEpochStepCallback(Callback):
    """`Callback` to plot the epoch number on the x axis instead of the step number."""

    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.log("step", float(trainer.current_epoch))
