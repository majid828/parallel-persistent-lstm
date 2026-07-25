import logging
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.training.parallel_sampler import (
    ParallelBasinSequenceBatchSampler
)


LOGGER = logging.getLogger(__name__)


class ParallelPersistentTrainer(BaseTrainer):
    """
    Parallel Persistent LSTM Trainer.

    Proposed strategy:

    Instead of:

        Basin A:
            seq1 -> seq2 -> seq3

    using one basin per batch,

    we train:

        Batch k:

        Basin A sequence k
        Basin B sequence k
        Basin C sequence k


    Each basin maintains an independent hidden/cell state:

        basin_id -> (hidden_state, cell_state)


    The original Persistent LSTM implementation remains unchanged.
    """


    def __init__(self, cfg):

        super().__init__(cfg)

        # Basin-specific hidden memory
        self.state_cache = {}



    # ==============================================================
    # INITIALIZATION
    # ==============================================================

    def initialize_training(self):

        """
        Initialize using BaseTrainer, then replace the
        standard loader with Parallel Persistent loader.
        """

        super().initialize_training()


        LOGGER.info(
            "### Parallel Persistent Trainer activated"
        )


        self.state_cache = {}


        if self.cfg.model.lower() != "persistentlstm":

            raise ValueError(
                "ParallelPersistentTrainer requires "
                "PersistentLSTM model."
            )


        self._initialize_parallel_sampler()



    # ==============================================================
    # PARALLEL SAMPLER
    # ==============================================================

    def _initialize_parallel_sampler(self):

        """
        Replace original Persistent LSTM sampler with
        ParallelBasinSequenceBatchSampler.
        """


        if not hasattr(
            self,
            "train_dataset"
        ):

            raise AttributeError(
                "BaseTrainer must expose train_dataset."
            )


        if not hasattr(
            self,
            "basin_to_sorted_indices"
        ):

            raise AttributeError(
                "BaseTrainer must expose "
                "basin_to_sorted_indices."
            )


        self._parallel_batch_sampler = (
            ParallelBasinSequenceBatchSampler(
                basin_to_sorted_indices=
                    self.basin_to_sorted_indices,

                n_basins_per_batch=
                    self.cfg.parallel_persistent_n_basins,

                drop_last=False,

                seed=
                    self.cfg.seed
                    if self.cfg.seed is not None
                    else 0,
            )
        )


        self.loader = DataLoader(

            self.train_dataset,

            batch_sampler=
                self._parallel_batch_sampler,

            num_workers=
                self.cfg.num_workers,

            pin_memory=
                self.cfg.pin_memory,

            collate_fn=
                self.train_dataset.collate_fn
        )


        LOGGER.info(
            "### Parallel Basin Sequence Sampler enabled"
        )



    # ==============================================================
    # HIDDEN STATE MANAGEMENT
    # ==============================================================

    def _get_hidden_state(self, basin_ids):

        """
        Collect hidden states according to current batch order.

        Parameters
        ----------
        basin_ids:
            Tensor [batch]

        Returns
        -------
        (h,c) or None
        """


        hidden_states = []
        cell_states = []


        # If no basin has previous history,
        # allow LSTM to initialize automatically.

        if all(
            int(b.item()) not in self.state_cache
            for b in basin_ids
        ):

            return None



        for basin in basin_ids:


            basin = int(
                basin.item()
            )


            if basin in self.state_cache:


                h,c = self.state_cache[basin]


                hidden_states.append(h)

                cell_states.append(c)


            else:

                # Missing basin state:
                # return None so LSTM starts from zero
                return None



        return (

            torch.cat(
                hidden_states,
                dim=1
            ),

            torch.cat(
                cell_states,
                dim=1
            )
        )



    def _update_hidden_state(
            self,
            basin_ids,
            hidden_state
    ):

        """
        Store hidden/cell state separately for each basin.
        """


        if hidden_state is None:

            return


        h,c = hidden_state



        for i, basin in enumerate(basin_ids):


            basin = int(
                basin.item()
            )


            self.state_cache[basin] = (

                h[:, i:i+1, :].detach(),

                c[:, i:i+1, :].detach()

            )



    # ==============================================================
    # TRAINING LOOP
    # ==============================================================

    def _train_epoch(self, epoch):


        self.model.train()


        self.experiment_logger.train()


        # Reset memory every epoch
        # Prevent information leakage between epochs

        self.state_cache = {}



        pbar = tqdm(

            self.loader,

            file=sys.stdout,

            disable=self._disable_pbar

        )


        pbar.set_description(
            f"# Epoch {epoch} Parallel Persistent"
        )


        nan_count = 0



        for _, data in enumerate(pbar):


            # ------------------------------------------------------
            # Move batch to device
            # ------------------------------------------------------

            for key in data.keys():


                if key.startswith("x_d"):


                    data[key] = {

                        k:v.to(self.device)

                        for k,v in data[key].items()

                    }


                elif not key.startswith("date"):


                    data[key] = data[key].to(
                        self.device
                    )



            data = self.model.pre_model_hook(
                data,
                is_train=True
            )



            # ------------------------------------------------------
            # Basin IDs
            # ------------------------------------------------------

            if "basin_idx" not in data:

                raise RuntimeError(
                    "Parallel Persistent LSTM requires basin_idx."
                )


            basin_ids = data["basin_idx"]


            if basin_ids.ndim > 1:

                basin_ids = basin_ids[:,0]



            # ------------------------------------------------------
            # Previous basin states
            # ------------------------------------------------------

            hidden_state = self._get_hidden_state(
                basin_ids
            )



            # ------------------------------------------------------
            # Forward pass
            # IMPORTANT:
            #
            # Keep multi basin structure.
            # No flattening.
            # ------------------------------------------------------

            predictions = self.model(

                data,

                hidden_state=hidden_state

            )



            # ------------------------------------------------------
            # Save states
            # ------------------------------------------------------

            self._update_hidden_state(

                basin_ids,

                predictions.get(
                    "hidden_state"
                )

            )



            # ------------------------------------------------------
            # Loss
            # ------------------------------------------------------

            loss_val, all_losses = self.loss_obj(
                predictions,
                data
            )



            if torch.isnan(loss_val):


                nan_count += 1


                if (
                    nan_count
                    >
                    self._allow_subsequent_nan_losses
                ):

                    raise RuntimeError(
                        "Loss NaN repeatedly."
                    )


                continue



            nan_count = 0



            # ------------------------------------------------------
            # Optimization
            # ------------------------------------------------------

            self.optimizer.zero_grad()


            loss_val.backward()



            if self.cfg.clip_gradient_norm is not None:


                torch.nn.utils.clip_grad_norm_(

                    self.model.parameters(),

                    self.cfg.clip_gradient_norm

                )



            self.optimizer.step()



            pbar.set_postfix_str(

                f"Loss {loss_val.item():.5f}"

            )


            self.experiment_logger.log_step(

                **{

                    k:v.item()

                    for k,v in all_losses.items()

                }

            )
