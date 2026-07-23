import random
from typing import Dict, List, Iterator

from torch.utils.data import Sampler


class ParallelBasinSequenceBatchSampler(Sampler[List[int]]):
    """
    Sampler for Parallel Persistent LSTM training.

    Creates heterogeneous multi-basin batches where the same temporal
    sequence index from different basins is processed together.

    Example
    -------
    Batch 1:
        Basin A sequence k
        Basin B sequence k
        Basin C sequence k


    Batch 2:
        Basin A sequence k+1
        Basin B sequence k+1
        Basin C sequence k+1


    Important properties
    --------------------
    1. Temporal ordering is preserved independently for each basin.

    2. Basin ordering inside each batch is randomly permuted.

       Example:

       Batch k:
           [A_k, C_k, B_k]

       Batch k+1:
           [B_k+1, A_k+1, C_k+1]


    3. Hidden/cell states must therefore be indexed by basin identity,
       not by batch position.
    """

    def __init__(
        self,
        basin_to_sorted_indices: Dict[int, List[int]],
        n_basins_per_batch: int,
        drop_last: bool = True,
        seed: int = 0,
    ):

        if len(basin_to_sorted_indices) == 0:
            raise ValueError(
                "basin_to_sorted_indices cannot be empty."
            )


        if n_basins_per_batch < 1:
            raise ValueError(
                "n_basins_per_batch must be >= 1."
            )


        self.basin_to_sorted_indices = (
            basin_to_sorted_indices
        )

        self.n_basins_per_batch = int(
            n_basins_per_batch
        )

        self.drop_last = drop_last

        self.seed = seed

        self._epoch = 0


        # ----------------------------------------------------------
        # Number of available sequences for every basin
        # ----------------------------------------------------------

        self.basin_lengths = {
            basin: len(indices)
            for basin, indices
            in basin_to_sorted_indices.items()
        }


        # All basins must contain at least one sequence
        if any(
            length < 1
            for length in self.basin_lengths.values()
        ):

            raise ValueError(
                "Every basin must contain at least one sequence."
            )


        # ----------------------------------------------------------
        # To synchronize sequence k across basins,
        # use the minimum available sequence length.
        # ----------------------------------------------------------

        self.max_sequence_length = min(
            self.basin_lengths.values()
        )


        self.basin_ids = list(
            self.basin_to_sorted_indices.keys()
        )


    def set_epoch(self, epoch: int):
        """
        Set epoch number for deterministic reshuffling.
        """

        self._epoch = epoch



    def __len__(self):
        """
        Number of optimizer updates per epoch.

        Each basin group produces:

            number_of_sequences

        batches.
        """

        if self.drop_last:

            n_groups = (
                len(self.basin_ids)
                //
                self.n_basins_per_batch
            )

        else:

            n_groups = (
                len(self.basin_ids)
                +
                self.n_basins_per_batch
                -
                1
            ) // self.n_basins_per_batch


        return (
            n_groups
            *
            self.max_sequence_length
        )



    def __iter__(self) -> Iterator[List[int]]:

        rng = random.Random(
            self.seed + self._epoch
        )


        # ----------------------------------------------------------
        # Shuffle basin groups every epoch
        # ----------------------------------------------------------

        basins = self.basin_ids.copy()

        rng.shuffle(basins)



        # ----------------------------------------------------------
        # Divide basins into groups
        # ----------------------------------------------------------

        for start in range(
            0,
            len(basins),
            self.n_basins_per_batch
        ):

            selected_basins = basins[
                start:
                start + self.n_basins_per_batch
            ]


            if (
                len(selected_basins)
                <
                self.n_basins_per_batch
                and self.drop_last
            ):
                continue



            # ------------------------------------------------------
            # Sequence synchronized training
            #
            # seq_idx is the temporal position.
            #
            # ------------------------------------------------------

            for seq_idx in range(
                self.max_sequence_length
            ):


                # --------------------------------------------------
                # Randomize basin ordering for this batch only
                # --------------------------------------------------

                batch_basins = (
                    selected_basins.copy()
                )

                rng.shuffle(
                    batch_basins
                )


                batch_indices = []


                for basin in batch_basins:


                    basin_sequences = (
                        self.basin_to_sorted_indices[
                            basin
                        ]
                    )


                    batch_indices.append(
                        basin_sequences[seq_idx]
                    )


                yield batch_indices
