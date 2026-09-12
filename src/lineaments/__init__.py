"""Parcours U-Net génériques pour les expériences ONHYM."""
from .config import experiences_disponibles, charger_configuration
from .experience import Experience

__version__ = "0.4.0"
__all__ = ["Experience", "experiences_disponibles", "charger_configuration"]
