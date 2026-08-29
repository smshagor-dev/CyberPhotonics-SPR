"""Qualification and hash-bound registration of physical research evidence."""

from sprpcf.evidence.qualification import (
    PHYSICAL_EVIDENCE_CLASSES,
    qualify_comsol_iteration,
    qualify_device_benchmark,
    qualify_experimental_sensor,
    validate_evidence_registry,
    write_evidence_record,
)

__all__ = [
    "PHYSICAL_EVIDENCE_CLASSES",
    "qualify_comsol_iteration",
    "qualify_device_benchmark",
    "qualify_experimental_sensor",
    "validate_evidence_registry",
    "write_evidence_record",
]
