from __future__ import annotations

import re
from dataclasses import dataclass

from dante.contracts import PrivacyClass


_SECRET = re.compile(r"(-----BEGIN [A-Z ]*PRIVATE KEY-----|\bpassword\s*=|\bapi[_-]?key\s*=|\bsk-[A-Za-z0-9_-]{12,})", re.IGNORECASE)
_CONFIDENTIAL = re.compile(r"(\bIBAN\b|\bfattur[ae]\b|\bcurriculum\b|\bcliente\b|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})", re.IGNORECASE)
_RANK = {value: index for index, value in enumerate(PrivacyClass)}


@dataclass(frozen=True)
class PrivacyDecision:
    classification: PrivacyClass
    cloud_allowed: bool
    model_allowed: bool
    reasons: tuple[str, ...]
    policy_version: str = "privacy-v0"


class PrivacyGate:
    def __init__(self, *, allow_internal_cloud: bool = True) -> None:
        self.allow_internal_cloud = allow_internal_cloud

    def classify(self, content: str, declared: PrivacyClass = PrivacyClass.INTERNAL) -> PrivacyDecision:
        detected = PrivacyClass.SECRET if _SECRET.search(content) else PrivacyClass.CONFIDENTIAL if _CONFIDENTIAL.search(content) else declared
        classification = max((declared, detected), key=lambda value: _RANK[value])
        if classification == PrivacyClass.SECRET:
            return PrivacyDecision(classification, False, False, ("secret_model_access_denied", "cloud_denied"))
        if classification == PrivacyClass.CONFIDENTIAL:
            return PrivacyDecision(classification, False, True, ("confidential_cloud_denied",))
        if classification == PrivacyClass.INTERNAL:
            return PrivacyDecision(classification, self.allow_internal_cloud, True, ("local_preferred", "cloud_policy_allowed" if self.allow_internal_cloud else "cloud_denied"))
        return PrivacyDecision(classification, True, True, ("public_cloud_allowed",))
