import random
from typing import Dict, List, Iterator

from torch.utils.data import Sampler


class ParallelBasinSequenceBatchSampler(Sampler[List[int]]):
    """
    Sampler for Parallel Persistent LSTM training.

    Purpose
    -------
    Creates heterogeneous batches where each batch contains the same
    sequence index from multiple basins.

    Example:

    Batch 1:
        Basin A seq 1
        Basin B seq 1
        Basin C seq 1

    Batch 2:
        Basin A seq 2
        Basin B seq 2
        Basin C seq 2


    This differs from the original Persistent LSTM sampler:

    Original:
        Basin A seq1
        Basin A seq2
        Basin A seq3


    Parallel:
        Seq k from all basins together.


    Requirements
    ------------
    The dataset indices must already be organized chronologically
    for every basin.
    """

    def __init__(
        self,
        basin_to_sorted_indices: Dict[int, List[int]],
        n_basins_per_batch: int,
        drop_last: bool = True,
        seed: int = 0,
    ):

        self.basin_to_sorted_indices = basin_to_sorted_indices
        self.n_basins_per_batch = int(n_basins_per_batch)
        self.drop_last = drop_last
        self.seed = seed

        self._epoch = 0


        # Number of sequences available per basin
        self.basin_lengths = {
            basin: len(indices)
            for basin, indices in basin_to_sorted_indices.items()
        }


        # Minimum length ensures all selected basins have
        # sequence k available
        self.max_sequence_length = min(
            self.basin_lengths.values()
        )


        # Basin IDs
        self.basin_ids = list(
            self.basin_to_sorted_indices.keys()
        )


    def set_epoch(self, epoch: int):
        """
        Change epoch for deterministic reshuffling.
        """

        self._epoch = epoch



    def __len__(self):

        if self.drop_last:

            return (
                len(self.basin_ids)
                //
                self.n_basins_per_batch
            )

        else:

            return (
                len(self.basin_ids)
                +
                self.n_basins_per_batch
                -
                1
            ) // self.n_basins_per_batch



def __iter__(self) -> Iterator[List[int]]:

    rng = random.Random(
        self.seed + self._epoch
    )


    # All basins available
    basins = self.basin_ids.copy()


    # Shuffle basin groups every epoch
    rng.shuffle(basins)


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



        # --------------------------------------------------
        # Sequence synchronized iteration
        # --------------------------------------------------

        for seq_idx in range(
            self.max_sequence_length
        ):


            # IMPORTANT:
            # Shuffle basin order INSIDE every batch
            #
            # Same sequence index,
            # different basin ordering
            #

            batch_basins = selected_basins.copy()

            rng.shuffle(batch_basins)


            batch_indices = []


            for basin in batch_basins:

                indices = (
                    self.basin_to_sorted_indices[basin]
                )


                batch_indices.append(
                    indices[seq_idx]
                )


            yield batch_indices
