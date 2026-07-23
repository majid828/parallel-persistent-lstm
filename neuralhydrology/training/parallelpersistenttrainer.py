import logging
import sys
from typing import Dict, Tuple, Optional

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
    Parallel Persistent LSTM trainer.

    Uses heterogeneous multi-basin batches:

        [
            Basin A sequence k,
            Basin B sequence k,
            Basin C sequence k
        ]

    while maintaining independent hidden/cell states:

        basin_id -> (hidden_state, cell_state)

    The original Persistent LSTM implementation remains
    unchanged through BaseTrainer.
    """


    def __init__(self, cfg):

        super().__init__(cfg)

        # Basin indexed hidden-state memory
        self.state_cache = {}



    # ==============================================================
    # TRAINING INITIALIZATION
    # ==============================================================

    def initialize_training(self):

        """
        Initialize training using BaseTrainer and replace
        the original persistent sampler with the parallel sampler.
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


        if not self.cfg.basin_state_cache:

            raise ValueError(
                "basin_state_cache must be enabled "
                "for Parallel Persistent LSTM."
            )


        self._initialize_parallel_sampler()



    def _initialize_parallel_sampler(self):

        """
        Replace the original Persistent LSTM sampler with
        ParallelBasinSequenceBatchSampler.
        """


        # ----------------------------------------------------------
        # Locate training dataset
        # ----------------------------------------------------------

        if hasattr(self, "dataset_train"):

            dataset = self.dataset_train

        elif hasattr(self, "train_dataset"):

            dataset = self.train_dataset

        elif hasattr(self, "dataset"):

            dataset = self.dataset

        else:

            raise AttributeError(
                "Training dataset not found in BaseTrainer."
            )


        # ----------------------------------------------------------
        # Locate basin chronological indices
        # ----------------------------------------------------------

        if hasattr(
            self,
            "basin_to_sorted_indices"
        ):

            basin_indices = (
                self.basin_to_sorted_indices
            )

        elif hasattr(
            self,
            "_basin_to_sorted_indices"
        ):

            basin_indices = (
                self._basin_to_sorted_indices
            )

        else:

            raise AttributeError(
                "Basin chronological indices not found."
            )


        sampler = ParallelBasinSequenceBatchSampler(

            basin_to_sorted_indices=basin_indices,

            n_basins_per_batch=
                self.cfg.parallel_persistent_n_basins,

            drop_last=True,

            seed=42
        )


        self.loader = DataLoader(

            dataset,

            batch_sampler=sampler,

            num_workers=self.cfg.num_workers,

            pin_memory=self.cfg.pin_memory
        )


        LOGGER.info(
            "Parallel basin sequence sampler enabled."
        )



    # ==============================================================
    # BASIN STATE MANAGEMENT
    # ==============================================================


    def _get_hidden_state(self, basin_ids):

        """
        Retrieve hidden/cell states using basin identity.

        Parameters
        ----------
        basin_ids:
            Tensor [number_of_basins]

        Returns
        -------
        tuple(h,c) or None
        """


        hidden_list = []
        cell_list = []


        # If at least one basin has no history,
        # initialize whole batch from zero state.
        #
        # LSTM will internally initialize missing states.

        for basin in basin_ids:


            basin = int(
                basin.item()
            )


            if basin not in self.state_cache:

                return None


            h,c = self.state_cache[basin]


            hidden_list.append(h)
            cell_list.append(c)



        return (

            torch.cat(
                hidden_list,
                dim=1
            ),

            torch.cat(
                cell_list,
                dim=1
            )

        )



    def _update_hidden_state(
            self,
            basin_ids,
            hidden_state
    ):

        """
        Save hidden/cell states separately for every basin.
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



        for i, data in enumerate(pbar):


            # ------------------------------------------------------
            # Move data to device
            # ------------------------------------------------------

            for key in data.keys():


                if key.startswith("x_d"):


                    data[key] = {

                        k: v.to(self.device)

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



            if basin_ids.ndim == 2:


                if not torch.all(

                    basin_ids == basin_ids[:,0:1]

                ):

                    raise RuntimeError(

                        "Multiple basin IDs detected inside "
                        "one sequence."

                    )


                basin_ids = basin_ids[:,0]



            # ------------------------------------------------------
            # Previous hidden state
            # ------------------------------------------------------

            hidden_state = self._get_hidden_state(
                basin_ids
            )



            # ------------------------------------------------------
            # Forward pass
            #
            # Keep multi-basin sequence structure.
            # No flattening.
            # ------------------------------------------------------

            predictions = self.model(

                data,

                hidden_state=hidden_state

            )



            # ------------------------------------------------------
            # Update basin states
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


                if nan_count > self._allow_subsequent_nan_losses:


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
