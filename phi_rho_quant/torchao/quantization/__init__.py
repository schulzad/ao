from typing import Any

import torch

try:
	from .prototype.phi_rho import PhiRhoCodebookConfig
	from .prototype.phi_rho.quantize_adapter import apply_phi_rho_to_module
except Exception:  # noqa: BLE001 - optional import in prototype
	PhiRhoCodebookConfig = None
	apply_phi_rho_to_module = None


def quantize_(module: torch.nn.Module, config: Any) -> torch.nn.Module:
	"""Minimal quantize_ router for the phi/rho prototype.

	In real TorchAO, this dispatches across many configs. Here we only handle
	PhiRhoCodebookConfig and otherwise return the module unchanged.
	"""
	if PhiRhoCodebookConfig is not None and isinstance(config, PhiRhoCodebookConfig):
		return apply_phi_rho_to_module(module, config)
	return module

__all__ = ["quantize_"]


