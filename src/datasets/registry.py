"""Registry pattern for dataset loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BaseNERDataset


class DatasetRegistry:
    """Registry for NER dataset loaders.

    This allows dynamic registration and retrieval of dataset loaders
    without hardcoding imports.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._loaders: dict[str, type[BaseNERDataset]] = {}
        self._register_default_loaders()

    def _register_default_loaders(self) -> None:
        """Register default dataset loaders."""
        # Import here to avoid circular imports
        from .loaders.ai4privacy import AI4PrivacyDataset
        from .loaders.bigcode_pii import BigCodePIIDataset
        from .loaders.conll import CoNLLDataset
        from .loaders.fewnerd import FewNERDDataset
        from .loaders.gretel_pii import GretelPIIDataset
        from .loaders.mendeley_pii import MendeleyPIIDataset
        from .loaders.ontonotes import OntoNotesDataset
        from .loaders.wikiner import WikiNERDataset
        from .loaders.wnut import WNUTDataset

        self.register("conll2003", CoNLLDataset)
        self.register("ontonotes", OntoNotesDataset)
        self.register("wnut17", WNUTDataset)
        self.register("fewnerd", FewNERDDataset)
        self.register("gretel_pii", GretelPIIDataset)
        self.register("wikiner", WikiNERDataset)
        self.register("ai4privacy", AI4PrivacyDataset)
        self.register("mendeley_pii", MendeleyPIIDataset)
        self.register("bigcode_pii", BigCodePIIDataset)

    def register(self, name: str, loader_class: type[BaseNERDataset]) -> None:
        """Register a dataset loader.

        Args:
            name: Name to register the loader under
            loader_class: Dataset loader class
        """
        self._loaders[name] = loader_class

    def get_loader(self, name: str, config: dict[str, Any] | None = None) -> BaseNERDataset:
        """Get a dataset loader instance.

        Args:
            name: Name of the dataset loader
            config: Optional configuration for the loader

        Returns:
            Initialized dataset loader

        Raises:
            ValueError: If loader not found
        """
        if name not in self._loaders:
            available = ", ".join(self._loaders.keys())
            raise ValueError(f"Unknown dataset: {name}. Available: {available}")

        loader_class = self._loaders[name]
        return loader_class(config)

    def list_available(self) -> list[str]:
        """List available dataset loaders.

        Returns:
            List of registered dataset names
        """
        return list(self._loaders.keys())
