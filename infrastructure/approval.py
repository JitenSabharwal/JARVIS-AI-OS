"""
Approval workflow manager for high-risk actions.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class ApprovalRequest:
    approval_id: str
    action: str
    requested_by: str
    reason: str
    resource: str = ""
    ttl_seconds: int = 900
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    approved_at: float | None = None
    approved_by: str = ""
    rejection_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    approval_token: str = field(default_factory=lambda: str(uuid.uuid4()))

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


class ApprovalManager:
    """Thread-safe in-memory approval manager singleton."""

    _instance: Optional["ApprovalManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._requests: Dict[str, ApprovalRequest] = {}
        self._token_index: Dict[str, str] = {}
        self._lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "ApprovalManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def create_request(
        self,
        *,
        action: str,
        requested_by: str,
        reason: str,
        resource: str = "",
        ttl_seconds: int = 900,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequest:
        approval = ApprovalRequest(
            approval_id=str(uuid.uuid4()),
            action=action,
            requested_by=requested_by,
            reason=reason,
            resource=resource,
            ttl_seconds=max(30, ttl_seconds),
            metadata=metadata or {},
        )
        with self._lock:
            self._requests[approval.approval_id] = approval
            self._token_index[approval.approval_token] = approval.approval_id
        return approval

    def get(self, approval_id: str) -> ApprovalRequest | None:
        with self._lock:
            approval = self._requests.get(approval_id)
            if approval is None:
                return None
            if approval.status == ApprovalStatus.PENDING and approval.is_expired():
                approval.status = ApprovalStatus.EXPIRED
            return approval

    def approve(self, approval_id: str, *, approver: str, note: str = "") -> ApprovalRequest | None:
        with self._lock:
            approval = self._requests.get(approval_id)
            if approval is None:
                return None
            if approval.is_expired():
                approval.status = ApprovalStatus.EXPIRED
                return approval
            approval.status = ApprovalStatus.APPROVED
            approval.approved_at = time.time()
            approval.approved_by = approver
            if note:
                approval.metadata["approval_note"] = note
            return approval

    def reject(self, approval_id: str, *, approver: str, reason: str = "") -> ApprovalRequest | None:
        with self._lock:
            approval = self._requests.get(approval_id)
            if approval is None:
                return None
            approval.status = ApprovalStatus.REJECTED
            approval.approved_at = time.time()
            approval.approved_by = approver
            approval.rejection_reason = reason
            return approval

    def validate_token(
        self,
        token: str,
        *,
        expected_action: str,
    ) -> bool:
        with self._lock:
            approval_id = self._token_index.get(token)
            if approval_id is None:
                return False
            approval = self._requests.get(approval_id)
            if approval is None:
                return False
            if approval.is_expired():
                approval.status = ApprovalStatus.EXPIRED
                return False
            if approval.status != ApprovalStatus.APPROVED:
                return False
            return approval.action == expected_action

