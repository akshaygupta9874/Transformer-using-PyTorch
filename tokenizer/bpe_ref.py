import heapq
from collections import Counter


class BPETrainer:
    """Deterministic BPE with lazy heap and local linked-list updates."""

    WORD_START = "▁"

    def __init__(self, vocab_size=40000, min_frequency=2, progress_every=100):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.progress_every = progress_every

        self.base_frequencies = Counter()
        self.base_vocabulary = set()
        self.atomic_vocabulary = set()

        self.merges = []
        self.merge_vocabulary = set()
        self.last_pair_frequency = 0

    # ------------------------------------------------------------------
    # Pair / boundary logic
    # ------------------------------------------------------------------

    @staticmethod
    def _boundary(unit):
        return (
            not unit
            or unit.isspace()
            or (
                len(unit) == 1
                and not unit.isalnum()
                and not any("\u0900" <= char <= "\u097f" for char in unit)
            )
        )

    @classmethod
    def _valid_pair(cls, left, right):
        if cls._boundary(left) or cls._boundary(right):
            return False

        # ▁ belongs to the following token and therefore must not
        # participate in a merge from the left side.
        if right.startswith(cls.WORD_START):
            return False

        return True

    @classmethod
    def _pairs(cls, sequence):
        for left, right in zip(sequence, sequence[1:]):
            if cls._valid_pair(left, right):
                yield left, right

    # ------------------------------------------------------------------
    # Deterministic heap
    # ------------------------------------------------------------------

    @staticmethod
    def _descending_text_key(value):
        return tuple(-ord(character) for character in value)

    @classmethod
    def _heap_entry(cls, pair, count):
        return (
            -count,
            cls._descending_text_key(pair[0]),
            cls._descending_text_key(pair[1]),
            pair[0],
            pair[1],
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        sequences,
        progress_callback=None,
        initialization_callback=None,
        preprocessing_callback=None,
    ):
        # ==============================================================
        # Phase 1: preprocessing
        # ==============================================================

        corpus = []

        for index, sequence in enumerate(sequences, 1):
            if isinstance(sequence, list):
                units = sequence
            else:
                units = list(sequence)

            corpus.append(units)

            if preprocessing_callback:
                preprocessing_callback(index)

        # ==============================================================
        # Phase 2: vocabulary
        # ==============================================================

        self.base_frequencies = Counter(
            unit
            for sequence in corpus
            for unit in sequence
        )

        self.base_vocabulary = set(self.base_frequencies)

        self.atomic_vocabulary = {
            character
            for sequence in corpus
            for unit in sequence
            for character in unit
        }

        vocabulary = set(self.base_vocabulary)

        # ==============================================================
        # Phase 3: build linked representation
        #
        # Each sequence becomes:
        #
        # token[i]
        # prev[i]
        # next[i]
        # alive[i]
        #
        # Nodes never move. A merge simply kills the right node.
        # ==============================================================

        token_values = []
        previous = []
        following = []
        alive = []

        for sequence in corpus:
            n = len(sequence)

            tokens = list(sequence)
            prev = [i - 1 for i in range(n)]
            nxt = [i + 1 for i in range(n)]

            if n:
                nxt[-1] = -1

            token_values.append(tokens)
            previous.append(prev)
            following.append(nxt)
            alive.append([True] * n)

        # We no longer need the original corpus.
        del corpus

        # ==============================================================
        # Phase 4: pair index
        #
        # pair_locations[pair] contains encoded node IDs.
        #
        # occurrence_id = sequence_id * stride + node_id
        #
        # This is substantially cheaper than storing:
        #     (sequence_index, node_index)
        # tuples.
        # ==============================================================

        max_nodes = max(
            (len(tokens) for tokens in token_values),
            default=0,
        )

        stride = max_nodes + 1

        pair_counts = Counter()
        pair_locations = {}

        total_sequences = len(token_values)

        for sequence_index, tokens in enumerate(token_values):

            if not tokens:
                if initialization_callback:
                    initialization_callback(
                        sequence_index + 1,
                        total_sequences,
                        len(pair_counts),
                    )
                continue

            nxt = following[sequence_index]

            for node_index in range(len(tokens) - 1):
                right_index = nxt[node_index]

                if right_index == -1:
                    continue

                left = tokens[node_index]
                right = tokens[right_index]

                if not self._valid_pair(left, right):
                    continue

                pair = (left, right)

                pair_counts[pair] += 1

                occurrence = sequence_index * stride + node_index

                locations = pair_locations.get(pair)

                if locations is None:
                    pair_locations[pair] = {occurrence}
                else:
                    locations.add(occurrence)

            if initialization_callback:
                initialization_callback(
                    sequence_index + 1,
                    total_sequences,
                    len(pair_counts),
                )

        # ==============================================================
        # Phase 5: initial heap
        # ==============================================================

        heap = [
            self._heap_entry(pair, count)
            for pair, count in pair_counts.items()
            if count >= self.min_frequency
        ]

        heapq.heapify(heap)

        if progress_callback:
            best_frequency = max(pair_counts.values(), default=0)

            progress_callback(
                0,
                len(vocabulary),
                best_frequency,
                self.vocab_size,
            )

        # ==============================================================
        # Local pair helpers
        # ==============================================================

        def remove_occurrence(pair, sequence_index, node_index):
            """Remove one pair occurrence from the index."""

            locations = pair_locations.get(pair)

            if locations is None:
                return

            occurrence = sequence_index * stride + node_index

            if occurrence not in locations:
                return

            locations.remove(occurrence)
            pair_counts[pair] -= 1

            if not locations:
                del pair_locations[pair]

        def add_occurrence(pair, sequence_index, node_index):
            """Add one pair occurrence to the index."""

            if pair_counts.get(pair, 0) == 0:
                locations = pair_locations.get(pair)

                if locations is None:
                    locations = set()
                    pair_locations[pair] = locations

            occurrence = sequence_index * stride + node_index

            locations = pair_locations[pair]

            if occurrence in locations:
                return

            locations.add(occurrence)
            pair_counts[pair] += 1

            heapq.heappush(
                heap,
                self._heap_entry(pair, pair_counts[pair]),
            )

        def remove_local_pair(sequence_index, left_index):
            """Remove pair beginning at left_index."""

            if left_index == -1:
                return

            if not alive[sequence_index][left_index]:
                return

            right_index = following[sequence_index][left_index]

            if right_index == -1:
                return

            if not alive[sequence_index][right_index]:
                return

            left = token_values[sequence_index][left_index]
            right = token_values[sequence_index][right_index]

            if not self._valid_pair(left, right):
                return

            remove_occurrence(
                (left, right),
                sequence_index,
                left_index,
            )

        def add_local_pair(sequence_index, left_index):
            """Add pair beginning at left_index."""

            if left_index == -1:
                return

            if not alive[sequence_index][left_index]:
                return

            right_index = following[sequence_index][left_index]

            if right_index == -1:
                return

            if not alive[sequence_index][right_index]:
                return

            left = token_values[sequence_index][left_index]
            right = token_values[sequence_index][right_index]

            if not self._valid_pair(left, right):
                return

            add_occurrence(
                (left, right),
                sequence_index,
                left_index,
            )

        # ==============================================================
        # Phase 6: BPE merging
        # ==============================================================

        while len(vocabulary) < self.vocab_size:

            pair = self._pop_current_pair(
                heap,
                pair_counts,
                vocabulary,
            )

            if pair is None:
                break

            current_frequency = pair_counts.get(pair, 0)

            if current_frequency < self.min_frequency:
                break

            self.last_pair_frequency = current_frequency

            left_token, right_token = pair
            merged = left_token + right_token

            # ----------------------------------------------------------
            # Register merge
            # ----------------------------------------------------------

            self.merges.append(pair)
            self.merge_vocabulary.add(merged)
            vocabulary.add(merged)

            # ----------------------------------------------------------
            # Get all current occurrences.
            #
            # Important: remove the entire entry first.
            # This handles overlapping occurrences correctly.
            # ----------------------------------------------------------

            occurrences = pair_locations.pop(pair, set())

            pair_counts.pop(pair, None)

            # ----------------------------------------------------------
            # Process only affected nodes.
            # ----------------------------------------------------------

            for occurrence in occurrences:

                sequence_index = occurrence // stride
                left_index = occurrence % stride

                if not alive[sequence_index][left_index]:
                    continue

                right_index = following[sequence_index][left_index]

                if right_index == -1:
                    continue

                if not alive[sequence_index][right_index]:
                    continue

                # Pair may have disappeared because of an earlier
                # overlapping merge.
                if (
                    token_values[sequence_index][left_index],
                    token_values[sequence_index][right_index],
                ) != pair:
                    continue

                prev_index = previous[sequence_index][left_index]
                next_index = following[sequence_index][right_index]

                # ------------------------------------------------------
                # Remove affected neighboring pairs BEFORE mutation.
                # ------------------------------------------------------

                remove_local_pair(
                    sequence_index,
                    prev_index,
                )

                # left -> right is the pair being merged.
                # It was already removed from pair_locations above.

                remove_local_pair(
                    sequence_index,
                    right_index,
                )

                # ------------------------------------------------------
                # Merge right node into left node.
                # ------------------------------------------------------

                token_values[sequence_index][left_index] = merged

                alive[sequence_index][right_index] = False

                following[sequence_index][left_index] = next_index

                if next_index != -1:
                    previous[sequence_index][next_index] = left_index

                # ------------------------------------------------------
                # Add newly-created neighboring pairs.
                # ------------------------------------------------------

                add_local_pair(
                    sequence_index,
                    prev_index,
                )

                add_local_pair(
                    sequence_index,
                    left_index,
                )

            # ----------------------------------------------------------
            # Progress
            # ----------------------------------------------------------

            merge_count = len(self.merges)

            if (
                progress_callback
                and merge_count % self.progress_every == 0
            ):
                progress_callback(
                    merge_count,
                    len(vocabulary),
                    self.last_pair_frequency,
                    self.vocab_size,
                )

        return self

    # ------------------------------------------------------------------
    # Heap
    # ------------------------------------------------------------------

    def _pop_current_pair(
        self,
        heap,
        pair_counts,
        vocabulary,
    ):
        while heap:

            negative_count, _, _, left, right = heapq.heappop(heap)

            pair = (left, right)
            count = -negative_count

            # Lazy deletion of stale heap entries.
            if pair_counts.get(pair, 0) != count:
                continue

            if count < self.min_frequency:
                continue

            merged = left + right

            if merged in vocabulary:
                continue

            return pair

        return None

    # ------------------------------------------------------------------
    # Apply trained merges
    # ------------------------------------------------------------------

    def apply(self, units):
        """Apply merges in rank order without scanning once per merge."""

        values = list(units)

        if not values or not self.merges:
            return values

        ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

        previous = [
            index - 1
            for index in range(len(values))
        ]

        following = [
            index + 1
            for index in range(len(values))
        ]

        if following:
            following[-1] = -1

        alive = [True] * len(values)

        heap = []

        def add_pair(left_index, right_index):
            if left_index < 0:
                return

            if right_index < 0:
                return

            if right_index >= len(values):
                return

            if not alive[left_index]:
                return

            if not alive[right_index]:
                return

            if following[left_index] != right_index:
                return

            pair = (
                values[left_index],
                values[right_index],
            )

            rank = ranks.get(pair)

            if rank is not None:
                heapq.heappush(
                    heap,
                    (
                        rank,
                        left_index,
                        right_index,
                        pair[0],
                        pair[1],
                    ),
                )

        for index in range(len(values) - 1):
            add_pair(index, index + 1)

        while heap:

            (
                _,
                left_index,
                right_index,
                left,
                right,
            ) = heapq.heappop(heap)

            if not alive[left_index]:
                continue

            if not alive[right_index]:
                continue

            if following[left_index] != right_index:
                continue

            if (
                values[left_index],
                values[right_index],
            ) != (left, right):
                continue

            values[left_index] = left + right

            alive[right_index] = False

            next_index = following[right_index]

            following[left_index] = next_index

            if next_index != -1:
                previous[next_index] = left_index

            add_pair(
                previous[left_index],
                left_index,
            )

            add_pair(
                left_index,
                next_index,
            )

        return [
            values[index]
            for index in range(len(values))
            if alive[index]
        ]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def vocab_statistics(self, budget=None):
        budget = budget or self.vocab_size

        base_count = len(self.base_vocabulary)

        return {
            "unique_base_units": base_count,
            "base_budget_fraction": base_count / budget,
            "remaining_merge_slots": max(
                0,
                budget - base_count - 4,
            ),
            "base_frequency": dict(self.base_frequencies),
        }