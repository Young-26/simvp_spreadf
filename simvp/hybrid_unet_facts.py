"""Backward-compatible imports for the SpreadF-STPN implementation.

New code should import from :mod:`simvp.spreadf_stpn`.
"""

from .spreadf_stpn import (
    FGST,
    FGSTB,
    FRRM,
    FactorizedGatedSpatiotemporalTransformer,
    FactorizedGatedSpatiotemporalTransformerBlock,
    FactorizedGatedSpatiotemporalTransformerStack,
    FRegionResidualRefinementModule,
    SpreadFSpatiotemporalPredictionNetwork,
    SpreadFSTPN,
)


HybridUNetFacTS = SpreadFSTPN
StrictFacTSTranslator = FactorizedGatedSpatiotemporalTransformer
GatedTransformerStack = FactorizedGatedSpatiotemporalTransformerStack
GatedTransformerBlock = FactorizedGatedSpatiotemporalTransformerBlock
LocalResidualRefiner = FRegionResidualRefinementModule
LocalFRegionBranch = FRegionResidualRefinementModule


__all__ = [
    "SpreadFSTPN",
    "SpreadFSpatiotemporalPredictionNetwork",
    "FactorizedGatedSpatiotemporalTransformer",
    "FactorizedGatedSpatiotemporalTransformerStack",
    "FactorizedGatedSpatiotemporalTransformerBlock",
    "FRegionResidualRefinementModule",
    "FGST",
    "FGSTB",
    "FRRM",
    "HybridUNetFacTS",
    "StrictFacTSTranslator",
    "GatedTransformerStack",
    "GatedTransformerBlock",
    "LocalResidualRefiner",
    "LocalFRegionBranch",
]
