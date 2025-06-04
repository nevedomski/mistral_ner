"""Dataset loaders for various NER datasets."""

from .ai4privacy import AI4PrivacyDataset
from .bigcode_pii import BigCodePIIDataset
from .conll import CoNLLDataset
from .fewnerd import FewNERDDataset
from .gretel_pii import GretelPIIDataset
from .mendeley_pii import MendeleyPIIDataset
from .ontonotes import OntoNotesDataset
from .wikiner import WikiNERDataset
from .wnut import WNUTDataset

__all__ = [
    "AI4PrivacyDataset",
    "BigCodePIIDataset",
    "CoNLLDataset",
    "FewNERDDataset",
    "GretelPIIDataset",
    "MendeleyPIIDataset",
    "OntoNotesDataset",
    "WNUTDataset",
    "WikiNERDataset",
]
