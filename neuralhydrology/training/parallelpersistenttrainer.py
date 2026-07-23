import logging
import sys
from typing import Dict, Tuple, Optional

import torch
from tqdm import tqdm

from neuralhydrology.training.basetrainer import BaseTrainer


LOGGER = logging.getLogger(__name__)


class ParallelPersistentTrainer(BaseTrainer):
    """
    Parallel Persistent LSTM trainer.

    Extends BaseTrainer while preserving:
        - optimizer
        - loss
        - validation
        - checkpointing
        - logging

    Difference from original Persistent LSTM:

    Original:
        one basin per batch

    Parallel:
        multiple basins per batch

        [
          basin1 sequence k,
          basin2 sequence k,
          basin3 sequence k
        ]

    Each basin owns an independent hidden/cell state.
    """

    def __init__(self, cfg):

        super().__init__(cfg)

        # basin_id -> (hidden_state, cell_state)
        self.state_cache = {}


    # ==============================================================
    # TRAINING INITIALIZATION
    # ==============================================================

    def initialize_training(self):

        """
        Use BaseTrainer initialization first.

        Then replace the loader with a parallel loader later.
        """

        super().initialize_training()

        LOGGER.info(
            "### Parallel Persistent Trainer activated"
        )


        # Reset basin memory
        self.state_cache = {}



    # ==============================================================
    # BASIN STATE MANAGEMENT
    # ==============================================================

    def _get_hidden_state(self, basin_ids):

        """
        Retrieve hidden states for each basin.

        basin_ids:
            tensor [B]

        Returns:
            h,c
        """

        hidden_list = []
        cell_list = []

        for basin in basin_ids:

            basin = int(basin.item())

            if basin in self.state_cache:

                h,c = self.state_cache[basin]

            else:

                h = None
                c = None


            hidden_list.append(h)
            cell_list.append(c)


        # First batch: no states available

        if all(h is None for h in hidden_list):
            return None


        return (
            torch.cat(hidden_list, dim=1),
            torch.cat(cell_list, dim=1)
        )


    def _update_hidden_state(
            self,
            basin_ids,
            hidden_state
    ):

        """
        Store hidden state separately for each basin.
        """

        if hidden_state is None:
            return


        h,c = hidden_state


        for i, basin in enumerate(basin_ids):

            basin = int(basin.item())


            self.state_cache[basin] = (
                h[:,i:i+1,:].detach(),
                c[:,i:i+1,:].detach()
            )


    # ==============================================================
    # PARALLEL TRAINING LOOP
    # ==============================================================

    def _train_epoch(self, epoch):

        self.model.train()

        self.experiment_logger.train()


        pbar = tqdm(
            self.loader,
            file=sys.stdout,
            disable=self._disable_pbar
        )


        pbar.set_description(
            f"# Epoch {epoch} Parallel Persistent"
        )


        nan_count = 0


        for i,data in enumerate(pbar):


            # ------------------------------------------------------
            # Move data to device
            # ------------------------------------------------------

            for key in data.keys():

                if key.startswith("x_d"):

                    data[key] = {
                        k:v.to(self.device)
                        for k,v in data[key].items()
                    }

                elif not key.startswith("date"):

                    data[key] = data[key].to(self.device)



            data = self.model.pre_model_hook(
                data,
                is_train=True
            )


            # ------------------------------------------------------
            # Basin IDs
            # ------------------------------------------------------

            if "basin_idx" not in data:

                raise RuntimeError(
                    "Parallel Persistent LSTM requires basin_idx"
                )


            basin_ids = data["basin_idx"]


            if basin_ids.dim() > 1:

                basin_ids = basin_ids[:,0]


            # ------------------------------------------------------
            # Get previous hidden states
            # ------------------------------------------------------

            hidden_state = self._get_hidden_state(
                basin_ids
            )


            # ------------------------------------------------------
            # Forward pass
            # IMPORTANT:
            # Keep [basin,time,features]
            # Do NOT flatten
            # ------------------------------------------------------

            predictions = self.model(
                data,
                hidden_state=hidden_state
            )


            # ------------------------------------------------------
            # Save new states
            # ------------------------------------------------------

            self._update_hidden_state(
                basin_ids,
                predictions.get("hidden_state")
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
