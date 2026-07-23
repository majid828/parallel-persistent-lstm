from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.utils.config import Config


def start_training(cfg: Config):
    """
    Start model training.

    The training strategy is selected through the config file.

    Options:

        training_strategy: homogeneous

            Uses the original NeuralHydrology BaseTrainer.
            This preserves the existing Persistent LSTM implementation.

        training_strategy: parallel

            Uses the new Parallel Persistent LSTM trainer.
    """

    # --------------------------------------------------------------
    # Select trainer according to training strategy
    # --------------------------------------------------------------

    if cfg.training_strategy == "homogeneous":

        # Existing Persistent LSTM training
        # No modification to previous implementation
        trainer = BaseTrainer(cfg=cfg)


    elif cfg.training_strategy == "parallel":

        # New Parallel Persistent LSTM training
        from neuralhydrology.training.parallelpersistenttrainer import (
            ParallelPersistentTrainer
        )

        trainer = ParallelPersistentTrainer(cfg=cfg)


    else:
        raise ValueError(
            f"Unknown training strategy: {cfg.training_strategy}"
        )


    trainer.initialize_training()
    trainer.train_and_validate()
