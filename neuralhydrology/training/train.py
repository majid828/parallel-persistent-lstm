from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.training.parallelpersistenttrainer import ParallelPersistentTrainer

from neuralhydrology.utils.config import Config



def start_training(cfg: Config):

    if cfg.head.lower() in ['regression', 'gmm', 'umal', 'cmal', '']:


        # =====================================================
        # NEW PARALLEL PERSISTENT LSTM SWITCH
        # =====================================================

        if getattr(cfg, "parallel_persistent", False):

            trainer = ParallelPersistentTrainer(cfg=cfg)

        else:

            trainer = BaseTrainer(cfg=cfg)


    else:

        raise ValueError(
            f"Unknown head {cfg.head}."
        )


    trainer.initialize_training()

    trainer.train_and_validate()
