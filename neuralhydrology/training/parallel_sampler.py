import logging
import random
from typing import Dict, Iterator, List

from torch.utils.data import Sampler


LOGGER = logging.getLogger(__name__)


class ParallelBasinSequenceBatchSampler(Sampler[List[int]]):
    """
    Batch sampler for Parallel Persistent LSTM training.

    At every temporal sequence position, the sampler includes all basins
    that still have an available sequence at that position.

    Example
    -------
    Suppose:

        Basin A has 2 sequences
        Basin B has 4 sequences
        Basin C has 3 sequences

    The sampler produces conceptually:

        Temporal position 0:
            [A_0, B_0, C_0]

        Temporal position 1:
            [A_1, B_1, C_1]

        Temporal position 2:
            [B_2, C_2]

        Temporal position 3:
            [B_3]

    Therefore, longer basins are not truncated to the length of the
    shortest basin.

    Important properties
    --------------------
    1. Temporal ordering is preserved independently for each basin.

    2. Basin order inside each batch is randomly shuffled.

    3. Hidden and cell states must be retrieved and stored using basin
       identity, not batch position.

    4. Shorter basins become inactive after their final sequence.

    5. Batch size can decrease near the end of an epoch.
    """

    def __init__(
        self,
        basin_to_sorted_indices: Dict[int, List[int]],
        n_basins_per_batch: int,
        drop_last: bool = False,
        seed: int = 0,
    ):
        """
        Initialize the sampler.

        Parameters
        ----------
        basin_to_sorted_indices
            Dictionary mapping each basin ID to a chronologically sorted
            list of dataset lookup-table indices.

        n_basins_per_batch
            Maximum number of basins included in one batch.

        drop_last
            If False, incomplete batches are retained. This should normally
            be False for parallel persistent training so that sequences from
            longer basins are not discarded after shorter basins finish.

            If True, an incomplete active-basin batch is discarded.

        seed
            Base random seed used for deterministic basin-order shuffling.
        """

        if not basin_to_sorted_indices:
            raise ValueError(
                "basin_to_sorted_indices cannot be empty."
            )

        if n_basins_per_batch < 1:
            raise ValueError(
                "n_basins_per_batch must be greater than or equal to 1."
            )

        self.basin_to_sorted_indices = basin_to_sorted_indices
        self.n_basins_per_batch = int(n_basins_per_batch)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)

        self._epoch = 0

        self.basin_ids = list(
            self.basin_to_sorted_indices.keys()
        )

        # ----------------------------------------------------------
        # Count available sequences for every basin
        # ----------------------------------------------------------
        self.basin_lengths = {
            basin: len(indices)
            for basin, indices
            in self.basin_to_sorted_indices.items()
        }

        empty_basins = [
            basin
            for basin, length in self.basin_lengths.items()
            if length < 1
        ]

        if empty_basins:
            raise ValueError(
                "Every basin must contain at least one sequence. "
                f"Basins with no available sequences: {empty_basins}"
            )

        # ----------------------------------------------------------
        # Continue until the longest basin is exhausted
        # ----------------------------------------------------------
        self.max_sequence_length = max(
            self.basin_lengths.values()
        )

        self.min_sequence_length = min(
            self.basin_lengths.values()
        )

        # Cache the exact sampler length because computing it repeatedly
        # can be unnecessary for a large number of basins.
        self._number_of_batches = self._calculate_number_of_batches()

        self._log_sequence_statistics()

    def _calculate_number_of_batches(self) -> int:
        """
        Calculate the exact number of batches produced per epoch.

        The number of active basins can decrease as temporal position
        increases, so sampler length cannot be calculated simply as:

            number_of_groups * maximum_sequence_length
        """

        number_of_batches = 0

        for sequence_position in range(
            self.max_sequence_length
        ):
            active_count = sum(
                1
                for basin in self.basin_ids
                if sequence_position < self.basin_lengths[basin]
            )

            if active_count == 0:
                continue

            if self.drop_last:
                number_of_batches += (
                    active_count
                    // self.n_basins_per_batch
                )
            else:
                number_of_batches += (
                    active_count
                    + self.n_basins_per_batch
                    - 1
                ) // self.n_basins_per_batch

        return number_of_batches

    def _log_sequence_statistics(self) -> None:
        """
        Log sequence-count diagnostics for all basins.
        """

        sorted_lengths = sorted(
            self.basin_lengths.items(),
            key=lambda item: (item[1], str(item[0])),
        )

        sequence_counts = list(
            self.basin_lengths.values()
        )

        minimum_count = min(sequence_counts)
        maximum_count = max(sequence_counts)
        mean_count = (
            sum(sequence_counts)
            / len(sequence_counts)
        )

        shortest_basins = [
            basin
            for basin, count in sorted_lengths
            if count == minimum_count
        ]

        longest_basins = [
            basin
            for basin, count in sorted_lengths
            if count == maximum_count
        ]

        basins_with_167 = [
            basin
            for basin, count in sorted_lengths
            if count == 167
        ]

        basins_with_exactly_3000 = [
            basin
            for basin, count in sorted_lengths
            if count == 3000
        ]

        basins_with_at_least_3000 = [
            basin
            for basin, count in sorted_lengths
            if count >= 3000
        ]

        total_available_sequences = sum(
            sequence_counts
        )

        LOGGER.info(
            "### Dynamic Parallel Persistent Sampler activated"
        )

        LOGGER.info(
            "### Total number of basins: %d",
            len(self.basin_ids),
        )

        LOGGER.info(
            "### Maximum basins per batch: %d",
            self.n_basins_per_batch,
        )

        LOGGER.info(
            "### drop_last: %s",
            self.drop_last,
        )

        LOGGER.info(
            "### Minimum sequences in one basin: %d",
            minimum_count,
        )

        LOGGER.info(
            "### Maximum sequences in one basin: %d",
            maximum_count,
        )

        LOGGER.info(
            "### Mean sequences per basin: %.2f",
            mean_count,
        )

        LOGGER.info(
            "### Total available basin-sequences: %d",
            total_available_sequences,
        )

        LOGGER.info(
            "### Dynamic temporal positions per epoch: %d",
            self.max_sequence_length,
        )

        LOGGER.info(
            "### Exact number of batches per epoch: %d",
            self._number_of_batches,
        )

        LOGGER.info(
            "### Basin(s) with the minimum sequence count: %s",
            shortest_basins,
        )

        LOGGER.info(
            "### Basin(s) with the maximum sequence count: %s",
            longest_basins,
        )

        # ----------------------------------------------------------
        # Basins with exactly 167 sequences
        # ----------------------------------------------------------
        LOGGER.info(
            "### Number of basins with exactly 167 sequences: %d",
            len(basins_with_167),
        )

        for basin in basins_with_167:
            LOGGER.info(
                "### 167-sequence basin: %s",
                str(basin),
            )

        # ----------------------------------------------------------
        # Basins with exactly 3000 sequences
        # ----------------------------------------------------------
        LOGGER.info(
            "### Number of basins with exactly 3000 sequences: %d",
            len(basins_with_exactly_3000),
        )

        for basin in basins_with_exactly_3000:
            LOGGER.info(
                "### 3000-sequence basin: %s",
                str(basin),
            )

        # ----------------------------------------------------------
        # Basins with at least 3000 sequences
        # ----------------------------------------------------------
        LOGGER.info(
            "### Number of basins with at least 3000 sequences: %d",
            len(basins_with_at_least_3000),
        )

        for basin in basins_with_at_least_3000:
            LOGGER.info(
                "### Basin with at least 3000 sequences: "
                "basin=%s, sequences=%d",
                str(basin),
                self.basin_lengths[basin],
            )

        # ----------------------------------------------------------
        # Twenty shortest basins
        # ----------------------------------------------------------
        LOGGER.info(
            "### 20 basins with the fewest available sequences:"
        )

        for rank, (basin, count) in enumerate(
            sorted_lengths[:20],
            start=1,
        ):
            LOGGER.info(
                "### Shortest basin rank %d: "
                "basin=%s, sequences=%d",
                rank,
                str(basin),
                count,
            )

        # ----------------------------------------------------------
        # Twenty longest basins
        # ----------------------------------------------------------
        LOGGER.info(
            "### 20 basins with the most available sequences:"
        )

        for rank, (basin, count) in enumerate(
            reversed(sorted_lengths[-20:]),
            start=1,
        ):
            LOGGER.info(
                "### Longest basin rank %d: "
                "basin=%s, sequences=%d",
                rank,
                str(basin),
                count,
            )

        if minimum_count != maximum_count:
            LOGGER.warning(
                "### Basin sequence counts are unequal. "
                "Dynamic active-basin sampling will be used. "
                "Short basins will become inactive after their final "
                "sequence, while longer basins will continue training."
            )

        if self.drop_last:
            LOGGER.warning(
                "### drop_last=True. Some remaining sequences may be "
                "discarded when the number of active basins becomes "
                "smaller than n_basins_per_batch. Use drop_last=False "
                "to train every available basin sequence."
            )

    def set_epoch(self, epoch: int) -> None:
        """
        Set the current epoch number.

        This creates deterministic but different basin shuffling for
        each epoch.
        """

        self._epoch = int(epoch)

    def __len__(self) -> int:
        """
        Return the exact number of batches generated in one epoch.
        """

        return self._number_of_batches

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield all available basin sequences.

        For each temporal sequence position:

        1. Find basins that still have an available sequence.
        2. Randomly shuffle the active basins.
        3. Divide active basins into one or more batches.
        4. Yield the corresponding dataset indices.

        Temporal order within each basin remains unchanged.
        """

        rng = random.Random(
            self.seed + self._epoch
        )

        # ----------------------------------------------------------
        # Move chronologically from sequence 0 to the final sequence
        # of the longest basin.
        # ----------------------------------------------------------
        for sequence_position in range(
            self.max_sequence_length
        ):
            # ------------------------------------------------------
            # Keep only basins that have data at this position.
            # ------------------------------------------------------
            active_basins = [
                basin
                for basin in self.basin_ids
                if (
                    sequence_position
                    < self.basin_lengths[basin]
                )
            ]

            if not active_basins:
                continue

            # ------------------------------------------------------
            # Randomize basin order at this temporal position.
            #
            # This changes basin position inside batches but does not
            # alter sequence chronology within any basin.
            # ------------------------------------------------------
            rng.shuffle(active_basins)

            # ------------------------------------------------------
            # Divide active basins into batches.
            # ------------------------------------------------------
            for start in range(
                0,
                len(active_basins),
                self.n_basins_per_batch,
            ):
                batch_basins = active_basins[
                    start:
                    start + self.n_basins_per_batch
                ]

                if (
                    len(batch_basins)
                    < self.n_basins_per_batch
                    and self.drop_last
                ):
                    continue

                batch_indices: List[int] = []

                for basin in batch_basins:
                    basin_sequences = (
                        self.basin_to_sorted_indices[
                            basin
                        ]
                    )

                    sequence_index = basin_sequences[
                        sequence_position
                    ]

                    batch_indices.append(
                        sequence_index
                    )

                if batch_indices:
                    yield batch_indices
