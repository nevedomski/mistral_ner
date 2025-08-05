"""Tag scheme converter for NER datasets.

This module provides functionality to detect and convert between different
NER tagging schemes (IO, IOB, IOB2, BIO, BIOES/BILOU).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import ClassVar

# Setup logging
logger = logging.getLogger(__name__)


class TagSchemeConverter:
    """Convert between different NER tagging schemes.

    Supported schemes:
    - IO: Inside-Outside (simplest scheme, no B- prefix)
    - IOB/IOB1: Inside-Outside-Beginning (B- only when needed)
    - IOB2/BIO: Inside-Outside-Beginning (B- for all entity starts)
    - BIOES/BILOU: Begin-Inside-Outside-End-Single / Begin-Inside-Last-Outside-Unit

    The converter can automatically detect the input scheme and convert to any
    supported output scheme.
    """

    # DOCDEV-NOTE: BIO and IOB2 are treated as equivalent - both use B- for all entity starts
    SUPPORTED_SCHEMES: ClassVar[list[str]] = ["IO", "IOB", "IOB2", "BIO", "BIOES", "BILOU"]

    @staticmethod
    def detect_scheme(tags: list[str]) -> str:
        """Detect the tagging scheme used in the given tags.

        Args:
            tags: List of NER tags from the dataset

        Returns:
            Detected scheme name (IO, IOB, IOB2/BIO, BIOES, BILOU)
        """
        # Count tag prefixes
        prefix_counts: Counter = Counter()
        has_entities = False

        for tag in tags:
            if tag == "O" or tag == "0":
                continue

            has_entities = True

            # Check for prefix
            if "-" in tag:
                prefix = tag.split("-")[0]
                prefix_counts[prefix] += 1
            else:
                # No prefix means IO scheme
                prefix_counts["IO"] += 1

        # If no entities found, default to BIO
        if not has_entities:
            return "BIO"

        # Check for IO scheme (no prefixes)
        if "IO" in prefix_counts and len(prefix_counts) == 1:
            return "IO"

        # Check for BIOES/BILOU schemes
        prefixes = set(prefix_counts.keys()) - {"IO"}

        if {"B", "I", "E", "S"}.issubset(prefixes) or {"B", "I", "O", "E", "S"}.issubset(prefixes):
            return "BIOES"
        elif {"B", "I", "L", "U"}.issubset(prefixes) or {"B", "I", "O", "L", "U"}.issubset(prefixes):
            return "BILOU"

        # Check for IOB vs IOB2/BIO
        # DOCDEV-QUESTION: IOB vs IOB2 detection is heuristic - may need sequence analysis for accuracy
        # For now, if we have B- and I- prefixes, assume BIO
        if {"B", "I"}.issubset(prefixes):
            return "BIO"
        elif "I" in prefixes:
            return "IOB"

        # Default to BIO
        return "BIO"

    @staticmethod
    def convert(tags: list[list[str]], from_scheme: str | None = None, to_scheme: str = "BIO") -> list[list[str]]:
        """Convert tags from one scheme to another.

        Args:
            tags: List of tag sequences (each sequence is a list of tags)
            from_scheme: Input scheme (if None, will auto-detect)
            to_scheme: Output scheme (default: BIO)

        Returns:
            Converted tag sequences
        """
        if to_scheme not in TagSchemeConverter.SUPPORTED_SCHEMES:
            raise ValueError(f"Unsupported output scheme: {to_scheme}. Supported: {TagSchemeConverter.SUPPORTED_SCHEMES}")

        # Flatten to detect scheme if needed
        if from_scheme is None:
            all_tags = [tag for seq in tags for tag in seq]
            from_scheme = TagSchemeConverter.detect_scheme(all_tags)
            logger.info(f"Detected tagging scheme: {from_scheme}")

        # If same scheme, return as-is
        # DOCDEV-NOTE: BIO and IOB2 are treated as equivalent
        if from_scheme == to_scheme or (from_scheme == "IOB2" and to_scheme == "BIO") or (from_scheme == "BIO" and to_scheme == "IOB2"):
            return tags

        # DOCDEV-TODO: POTENTIAL INFINITE LOOP RISK - Chained lambda conversions
        # Risk: Application hang if circular conversion dependencies exist
        # Add conversion cycle detection and maximum depth limits
        # Convert based on scheme pair
        converter_map = {
            ("IO", "BIO"): TagSchemeConverter._io_to_bio,
            ("IO", "BIOES"): lambda tags: TagSchemeConverter._bio_to_bioes(TagSchemeConverter._io_to_bio(tags)),
            ("IO", "BILOU"): lambda tags: TagSchemeConverter._bio_to_bilou(TagSchemeConverter._io_to_bio(tags)),
            ("IOB", "BIO"): TagSchemeConverter._iob_to_bio,
            ("IOB", "BIOES"): lambda tags: TagSchemeConverter._bio_to_bioes(TagSchemeConverter._iob_to_bio(tags)),
            ("IOB", "BILOU"): lambda tags: TagSchemeConverter._bio_to_bilou(TagSchemeConverter._iob_to_bio(tags)),
            ("BIO", "IO"): TagSchemeConverter._bio_to_io,
            ("BIO", "BIOES"): TagSchemeConverter._bio_to_bioes,
            ("BIO", "BILOU"): TagSchemeConverter._bio_to_bilou,
            ("IOB2", "IO"): TagSchemeConverter._bio_to_io,
            ("IOB2", "BIOES"): TagSchemeConverter._bio_to_bioes,
            ("IOB2", "BILOU"): TagSchemeConverter._bio_to_bilou,
            ("BIOES", "BIO"): TagSchemeConverter._bioes_to_bio,
            ("BIOES", "IO"): lambda tags: TagSchemeConverter._bio_to_io(TagSchemeConverter._bioes_to_bio(tags)),
            ("BIOES", "BILOU"): lambda tags: TagSchemeConverter._bio_to_bilou(TagSchemeConverter._bioes_to_bio(tags)),
            ("BILOU", "BIO"): TagSchemeConverter._bilou_to_bio,
            ("BILOU", "IO"): lambda tags: TagSchemeConverter._bio_to_io(TagSchemeConverter._bilou_to_bio(tags)),
            ("BILOU", "BIOES"): lambda tags: TagSchemeConverter._bio_to_bioes(TagSchemeConverter._bilou_to_bio(tags)),
        }

        converter_key = (from_scheme, to_scheme)
        if converter_key in converter_map:
            logger.info(f"Converting from {from_scheme} to {to_scheme}")
            return converter_map[converter_key](tags)
        else:
            raise ValueError(f"Conversion from {from_scheme} to {to_scheme} not supported")

    @staticmethod
    def _io_to_bio(tags: list[list[str]]) -> list[list[str]]:
        """Convert IO scheme to BIO scheme."""
        converted = []

        for sequence in tags:
            new_sequence = []
            prev_label = "O"

            for tag in sequence:
                if tag == "O" or tag == "0":
                    new_sequence.append("O")
                    prev_label = "O"
                else:
                    # Entity tag without prefix
                    # DOCDEV-NOTE: IO scheme detection based on tags without B-/I- prefixes
                    if prev_label == "O" or prev_label != tag:
                        # Start of new entity
                        new_sequence.append(f"B-{tag}")
                    else:
                        # Continuation of entity
                        new_sequence.append(f"I-{tag}")
                    prev_label = tag

            converted.append(new_sequence)

        return converted

    @staticmethod
    def _iob_to_bio(tags: list[list[str]]) -> list[list[str]]:
        """Convert IOB scheme to BIO scheme."""
        converted = []

        for sequence in tags:
            new_sequence = []
            prev_tag = "O"

            for tag in sequence:
                if tag == "O":
                    new_sequence.append("O")
                    prev_tag = "O"
                elif tag.startswith("B-"):
                    new_sequence.append(tag)
                    prev_tag = tag
                elif tag.startswith("I-"):
                    # Check if this should be B- instead
                    # DOCDEV-NOTE: IOB1 allows I- to start entities, BIO/IOB2 requires B- for all starts
                    entity_type = tag[2:]
                    prev_entity_type = prev_tag[2:] if prev_tag != "O" else None

                    if prev_tag == "O" or entity_type != prev_entity_type:
                        # Start of new entity
                        new_sequence.append(f"B-{entity_type}")
                    else:
                        # Continuation
                        new_sequence.append(tag)
                    prev_tag = tag
                else:
                    # Handle tags without prefixes
                    new_sequence.append(tag)
                    prev_tag = tag

            converted.append(new_sequence)

        return converted

    @staticmethod
    def _bio_to_io(tags: list[list[str]]) -> list[list[str]]:
        """Convert BIO scheme to IO scheme."""
        converted = []

        for sequence in tags:
            new_sequence = []

            for tag in sequence:
                if tag == "O":
                    new_sequence.append("O")
                elif tag.startswith("B-") or tag.startswith("I-"):
                    # Remove prefix
                    entity_type = tag[2:]
                    new_sequence.append(entity_type)
                else:
                    new_sequence.append(tag)

            converted.append(new_sequence)

        return converted

    @staticmethod
    def _bio_to_bioes(tags: list[list[str]]) -> list[list[str]]:
        """Convert BIO scheme to BIOES scheme."""
        converted = []

        for sequence in tags:
            new_sequence = []

            for i, tag in enumerate(sequence):
                if tag == "O":
                    new_sequence.append("O")
                elif tag.startswith("B-"):
                    entity_type = tag[2:]
                    # Check if this is a single-token entity
                    # DOCDEV-NOTE: S- prefix for single-token entities in BIOES scheme
                    if i + 1 >= len(sequence) or not sequence[i + 1].startswith(f"I-{entity_type}"):
                        new_sequence.append(f"S-{entity_type}")
                    else:
                        new_sequence.append(tag)
                elif tag.startswith("I-"):
                    entity_type = tag[2:]
                    # Check if this is the last token of the entity
                    if i + 1 >= len(sequence) or not sequence[i + 1].startswith(f"I-{entity_type}"):
                        new_sequence.append(f"E-{entity_type}")
                    else:
                        new_sequence.append(tag)
                else:
                    new_sequence.append(tag)

            converted.append(new_sequence)

        return converted

    @staticmethod
    def _bio_to_bilou(tags: list[list[str]]) -> list[list[str]]:
        """Convert BIO scheme to BILOU scheme."""
        converted = []

        for sequence in tags:
            new_sequence = []

            for i, tag in enumerate(sequence):
                if tag == "O":
                    new_sequence.append("O")
                elif tag.startswith("B-"):
                    entity_type = tag[2:]
                    # Check if this is a single-token entity
                    # DOCDEV-NOTE: U- prefix for unit (single-token) entities in BILOU scheme
                    if i + 1 >= len(sequence) or not sequence[i + 1].startswith(f"I-{entity_type}"):
                        new_sequence.append(f"U-{entity_type}")
                    else:
                        new_sequence.append(tag)
                elif tag.startswith("I-"):
                    entity_type = tag[2:]
                    # Check if this is the last token of the entity
                    if i + 1 >= len(sequence) or not sequence[i + 1].startswith(f"I-{entity_type}"):
                        new_sequence.append(f"L-{entity_type}")
                    else:
                        new_sequence.append(tag)
                else:
                    new_sequence.append(tag)

            converted.append(new_sequence)

        return converted

    @staticmethod
    def _bioes_to_bio(tags: list[list[str]]) -> list[list[str]]:
        """Convert BIOES scheme to BIO scheme."""
        converted = []

        for sequence in tags:
            new_sequence = []

            for tag in sequence:
                if tag == "O":
                    new_sequence.append("O")
                elif tag.startswith("S-"):
                    # Single -> Begin
                    entity_type = tag[2:]
                    new_sequence.append(f"B-{entity_type}")
                elif tag.startswith("E-"):
                    # End -> Inside
                    entity_type = tag[2:]
                    new_sequence.append(f"I-{entity_type}")
                elif tag.startswith("B-") or tag.startswith("I-"):
                    # Keep as-is
                    new_sequence.append(tag)
                else:
                    new_sequence.append(tag)

            converted.append(new_sequence)

        return converted

    @staticmethod
    def _bilou_to_bio(tags: list[list[str]]) -> list[list[str]]:
        """Convert BILOU scheme to BIO scheme."""
        converted = []

        for sequence in tags:
            new_sequence = []

            for tag in sequence:
                if tag == "O":
                    new_sequence.append("O")
                elif tag.startswith("U-"):
                    # Unit -> Begin
                    entity_type = tag[2:]
                    new_sequence.append(f"B-{entity_type}")
                elif tag.startswith("L-"):
                    # Last -> Inside
                    entity_type = tag[2:]
                    new_sequence.append(f"I-{entity_type}")
                elif tag.startswith("B-") or tag.startswith("I-"):
                    # Keep as-is
                    new_sequence.append(tag)
                else:
                    new_sequence.append(tag)

            converted.append(new_sequence)

        return converted
