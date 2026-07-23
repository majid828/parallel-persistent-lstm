from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.training.parallelpersistenttrainer import (
    ParallelPersistentTrainer
)

from neuralhydrology.utils.config import Config



def start_training(cfg: Config):
    """
    Start model training.

    Trainer selection is controlled through the config:

    training_strategy: homogeneous
        -> Original Persistent LSTM / BaseTrainer

    training_strategy: parallel
        -> Proposed Parallel Persistent LSTM

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """


    # MC-LSTM is a special case, where the head returns an empty string
    # but the model is trained as regression model.

    if cfg.head.lower() in [
        'regression',
        'gmm',
        'umal',
        'cmal',
        ''
    ]:


        # ==========================================================
        # Trainer selection
        # ==========================================================

        if cfg.parallel_persistent:

            trainer = ParallelPersistentTrainer(
                cfg=cfg
            )

        else:

            trainer = BaseTrainer(
                cfg=cfg
            )


    else:

        raise ValueError(
            f"Unknown head {cfg.head}."
        )



    # ==========================================================
    # Start training
    # ==========================================================

    trainer.initialize_training()

    trainer.train_and_validate()
