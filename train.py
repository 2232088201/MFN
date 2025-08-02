from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI

from mfn.datamodule import HMEDatamodule
from mfn.lit_mfn import LitMFN

cli = LightningCLI(
    LitMFN,
    HMEDatamodule,
    save_config_overwrite=True,
    trainer_defaults={"plugins": DDPPlugin(find_unused_parameters=False)},
)
