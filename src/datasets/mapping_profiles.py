"""Predefined label mapping profiles for common use cases."""

from __future__ import annotations

from typing import ClassVar


class MappingProfiles:
    """Container for predefined label mapping profiles."""

    # Bank PII profile - maps all location types to ADDR, drops irrelevant entities
    BANK_PII: ClassVar[dict[str, dict[str, str]]] = {
        "conll2003": {
            "O": "O",
            "B-PER": "B-PER",
            "I-PER": "I-PER",
            "B-ORG": "B-ORG",
            "I-ORG": "I-ORG",
            "B-LOC": "B-ADDR",  # Map locations to addresses
            "I-LOC": "I-ADDR",
            "B-MISC": "B-MISC",
            "I-MISC": "I-MISC",
        },
        "ontonotes": {
            "O": "O",
            # Person
            "B-PERSON": "B-PER",
            "I-PERSON": "I-PER",
            # Organizations
            "B-ORG": "B-ORG",
            "I-ORG": "I-ORG",
            # Locations - all map to ADDR for bank PII
            "B-LOC": "B-ADDR",
            "I-LOC": "I-ADDR",
            "B-GPE": "B-ADDR",  # Cities, countries -> address info
            "I-GPE": "I-ADDR",
            "B-FAC": "B-ADDR",  # Facilities -> address info
            "I-FAC": "I-ADDR",
            # Temporal
            "B-DATE": "B-DATE",
            "I-DATE": "I-DATE",
            "B-TIME": "B-TIME",
            "I-TIME": "I-TIME",
            # Numeric
            "B-MONEY": "B-MONEY",
            "I-MONEY": "I-MONEY",
            "B-PERCENT": "B-PERCENT",
            "I-PERCENT": "I-PERCENT",
            "B-QUANTITY": "O",  # Not relevant for bank PII
            "I-QUANTITY": "O",
            "B-ORDINAL": "O",  # Not relevant for bank PII
            "I-ORDINAL": "O",
            "B-CARDINAL": "O",  # Not relevant for bank PII
            "I-CARDINAL": "O",
            # Other
            "B-NORP": "B-MISC",  # Nationalities -> misc PII
            "I-NORP": "I-MISC",
            "B-PRODUCT": "B-MISC",  # Products -> misc
            "I-PRODUCT": "I-MISC",
            "B-EVENT": "O",  # Not relevant for bank PII
            "I-EVENT": "O",
            "B-WORK_OF_ART": "O",  # Not relevant for bank PII
            "I-WORK_OF_ART": "O",
            "B-LAW": "O",  # Not relevant for bank PII
            "I-LAW": "O",
            "B-LANGUAGE": "O",  # Not relevant for bank PII
            "I-LANGUAGE": "O",
        },
        "gretel_pii": {
            "O": "O",
            # Personal identifiers
            "B-PERSON": "B-PER",
            "I-PERSON": "I-PER",
            "B-NAME": "B-PER",
            "I-NAME": "I-PER",
            # Financial
            "B-CREDIT_CARD": "B-CARD",
            "I-CREDIT_CARD": "I-CARD",
            "B-BANK_ACCOUNT": "B-BANK",
            "I-BANK_ACCOUNT": "I-BANK",
            "B-IBAN": "B-BANK",
            "I-IBAN": "I-BANK",
            "B-SWIFT": "B-BANK",
            "I-SWIFT": "I-BANK",
            "B-ROUTING_NUMBER": "B-BANK",
            "I-ROUTING_NUMBER": "I-BANK",
            "B-ACCOUNT_NUMBER": "B-BANK",
            "I-ACCOUNT_NUMBER": "I-BANK",
            # Government IDs
            "B-SSN": "B-SSN",
            "I-SSN": "I-SSN",
            "B-PASSPORT": "B-PASSPORT",
            "I-PASSPORT": "I-PASSPORT",
            "B-LICENSE": "B-LICENSE",
            "I-LICENSE": "I-LICENSE",
            "B-DRIVER_LICENSE": "B-LICENSE",
            "I-DRIVER_LICENSE": "I-LICENSE",
            # Contact
            "B-PHONE": "B-PHONE",
            "I-PHONE": "I-PHONE",
            "B-EMAIL": "B-EMAIL",
            "I-EMAIL": "I-EMAIL",
            "B-ADDRESS": "B-ADDR",
            "I-ADDRESS": "I-ADDR",
            # Locations -> ADDR for bank PII
            "B-CITY": "B-ADDR",
            "I-CITY": "I-ADDR",
            "B-STATE": "B-ADDR",
            "I-STATE": "I-ADDR",
            "B-COUNTRY": "B-ADDR",
            "I-COUNTRY": "I-ADDR",
            "B-ZIPCODE": "B-ADDR",
            "I-ZIPCODE": "I-ADDR",
            # Dates and numbers
            "B-DATE": "B-DATE",
            "I-DATE": "I-DATE",
            "B-DOB": "B-DATE",  # Date of birth -> DATE
            "I-DOB": "I-DATE",
            "B-DATE_OF_BIRTH": "B-DATE",
            "I-DATE_OF_BIRTH": "I-DATE",
            # Organizations
            "B-COMPANY": "B-ORG",
            "I-COMPANY": "I-ORG",
            "B-ORGANIZATION": "B-ORG",
            "I-ORGANIZATION": "I-ORG",
            # Monetary values
            "B-AMOUNT": "B-MONEY",
            "I-AMOUNT": "I-MONEY",
            "B-CURRENCY": "B-MONEY",
            "I-CURRENCY": "I-MONEY",
            # Generic PII
            "B-PII": "B-MISC",
            "I-PII": "I-MISC",
            "B-USERNAME": "B-MISC",
            "I-USERNAME": "I-MISC",
            "B-PASSWORD": "B-MISC",
            "I-PASSWORD": "I-MISC",
            "B-IP_ADDRESS": "B-MISC",
            "I-IP_ADDRESS": "I-MISC",
        },
        "ai4privacy": {
            "O": "O",
            # Personal names
            "B-PREFIX": "B-PER",
            "I-PREFIX": "I-PER",
            "B-FIRSTNAME": "B-PER",
            "I-FIRSTNAME": "I-PER",
            "B-MIDDLENAME": "B-PER",
            "I-MIDDLENAME": "I-PER",
            "B-LASTNAME": "B-PER",
            "I-LASTNAME": "I-PER",
            "B-NAME": "B-PER",
            "I-NAME": "I-PER",
            # Organizations
            "B-COMPANY_NAME": "B-ORG",
            "I-COMPANY_NAME": "I-ORG",
            "B-COMPANYNAME": "B-ORG",
            "I-COMPANYNAME": "I-ORG",
            # Locations - all map to ADDR for bank PII
            "B-CITY": "B-ADDR",
            "I-CITY": "I-ADDR",
            "B-STATE": "B-ADDR",
            "I-STATE": "I-ADDR",
            "B-COUNTRY": "B-ADDR",
            "I-COUNTRY": "I-ADDR",
            "B-STREET": "B-ADDR",
            "I-STREET": "I-ADDR",
            "B-BUILDINGNUMBER": "B-ADDR",
            "I-BUILDINGNUMBER": "I-ADDR",
            "B-ZIPCODE": "B-ADDR",
            "I-ZIPCODE": "I-ADDR",
            "B-SECONDARYADDRESS": "B-ADDR",
            "I-SECONDARYADDRESS": "I-ADDR",
            # Financial
            "B-CREDITCARDNUMBER": "B-CARD",
            "I-CREDITCARDNUMBER": "I-CARD",
            "B-CREDITCARDCVV": "B-CARD",
            "I-CREDITCARDCVV": "I-CARD",
            "B-CREDITCARDISSUER": "B-ORG",
            "I-CREDITCARDISSUER": "I-ORG",
            "B-ACCOUNTNUMBER": "B-BANK",
            "I-ACCOUNTNUMBER": "I-BANK",
            "B-IBAN": "B-BANK",
            "I-IBAN": "I-BANK",
            "B-BIC": "B-BANK",
            "I-BIC": "I-BANK",
            "B-AMOUNT": "B-MONEY",
            "I-AMOUNT": "I-MONEY",
            "B-CURRENCY": "B-MONEY",
            "I-CURRENCY": "I-MONEY",
            # Contact info
            "B-EMAIL": "B-EMAIL",
            "I-EMAIL": "I-EMAIL",
            "B-PHONEIMEI": "B-PHONE",
            "I-PHONEIMEI": "I-PHONE",
            "B-PHONENUMBER": "B-PHONE",
            "I-PHONENUMBER": "I-PHONE",
            # IDs
            "B-SSN": "B-SSN",
            "I-SSN": "I-SSN",
            "B-DRIVERLICENSE": "B-LICENSE",
            "I-DRIVERLICENSE": "I-LICENSE",
            "B-PASSPORT": "B-PASSPORT",
            "I-PASSPORT": "I-PASSPORT",
            "B-IDCARD": "B-MISC",
            "I-IDCARD": "I-MISC",
            "B-VEHICLEIDENTIFICATIONNUMBER": "B-MISC",
            "I-VEHICLEIDENTIFICATIONNUMBER": "I-MISC",
            "B-VEHICLEREGISTRATION": "B-MISC",
            "I-VEHICLEREGISTRATION": "I-MISC",
            # Dates and times
            "B-DATE": "B-DATE",
            "I-DATE": "I-DATE",
            "B-TIME": "B-TIME",
            "I-TIME": "I-TIME",
            "B-DOB": "B-DATE",
            "I-DOB": "I-DATE",
            "B-AGE": "B-MISC",
            "I-AGE": "I-MISC",
            # Medical
            "B-MEDICALLICENSE": "B-LICENSE",
            "I-MEDICALLICENSE": "I-LICENSE",
            "B-ACCOUNTNAME": "B-PER",
            "I-ACCOUNTNAME": "I-PER",
            # Other
            "B-USERNAME": "B-MISC",
            "I-USERNAME": "I-MISC",
            "B-PASSWORD": "B-MISC",
            "I-PASSWORD": "I-MISC",
            "B-IP": "B-MISC",
            "I-IP": "I-MISC",
            "B-IPV4": "B-MISC",
            "I-IPV4": "I-MISC",
            "B-IPV6": "B-MISC",
            "I-IPV6": "I-MISC",
            "B-MAC": "B-MISC",
            "I-MAC": "I-MISC",
            "B-URL": "B-MISC",
            "I-URL": "I-MISC",
            "B-JOBAREA": "B-MISC",
            "I-JOBAREA": "I-MISC",
            "B-JOBTITLE": "B-MISC",
            "I-JOBTITLE": "I-MISC",
            "B-JOBDESCRIPTOR": "B-MISC",
            "I-JOBDESCRIPTOR": "I-MISC",
            "B-GENDER": "B-MISC",
            "I-GENDER": "I-MISC",
            "B-SEX": "B-MISC",
            "I-SEX": "I-MISC",
            "B-USERAGENT": "B-MISC",
            "I-USERAGENT": "I-MISC",
            "B-MASKEDNUMBER": "B-MISC",
            "I-MASKEDNUMBER": "I-MISC",
            "B-PIN": "B-MISC",
            "I-PIN": "I-MISC",
        },
    }

    # General profile - preserves most entity distinctions
    GENERAL: ClassVar[dict[str, dict[str, str]]] = {
        "conll2003": {
            # Identity mapping for CoNLL-2003
            "O": "O",
            "B-PER": "B-PER",
            "I-PER": "I-PER",
            "B-ORG": "B-ORG",
            "I-ORG": "I-ORG",
            "B-LOC": "B-LOC",
            "I-LOC": "I-LOC",
            "B-MISC": "B-MISC",
            "I-MISC": "I-MISC",
        },
        # Add other datasets as needed
    }

    @classmethod
    def get_profile(cls, profile_name: str) -> dict[str, dict[str, str]]:
        """Get a mapping profile by name.

        Args:
            profile_name: Name of the profile (e.g., "bank_pii", "general")

        Returns:
            Dictionary of dataset mappings

        Raises:
            ValueError: If profile not found
        """
        profile_name_upper = profile_name.upper()
        if hasattr(cls, profile_name_upper):
            profile = getattr(cls, profile_name_upper)
            assert isinstance(profile, dict)  # Type narrowing for mypy
            return profile

        available = [attr for attr in dir(cls) if attr.isupper() and not attr.startswith("_")]
        raise ValueError(f"Unknown profile: {profile_name}. Available: {available}")

    @classmethod
    def list_profiles(cls) -> list[str]:
        """List available mapping profiles.

        Returns:
            List of profile names
        """
        return [attr.lower() for attr in dir(cls) if attr.isupper() and not attr.startswith("_")]
