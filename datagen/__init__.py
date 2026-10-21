"""Synthetic dataset generator for The Catcher in the RAG."""

from datagen.generator import generate_base_package, generate_dataset, generate_package
from datagen.injector import inject_anomaly

__all__ = [
    "generate_base_package",
    "generate_dataset",
    "generate_package",
    "inject_anomaly",
]
