from .convlstm_model import ConvLSTM_Model
from .earthfarseer_model import EarthFarseer_Model
from .hybrid_unet_facts import HybridUNetFacTS, LocalFRegionBranch, LocalResidualRefiner
from .mau_model import MAU_Model
from .mim_model import MIM_Model
from .model import SimVP
from .predrnnpp_model import PredRNNpp_Model
from .predrnnv2_model import PredRNNv2_Model
from .tau_model import TAU_Model
from .wrapper import SUPPORTED_ARCHS, SimVPForecast

try:
    from .predformer_facts_model import PredFormerFacTS_Model
    from .predformer_quadruplet_tsst_model import PredFormerQuadrupletTSST_Model
except ImportError:
    # Keep package import usable for non-PredFormer architectures even when the
    # local PredFormer files are out of sync. wrapper.py raises a targeted error
    # if a PredFormer arch is actually requested.
    PredFormerFacTS_Model = None
    PredFormerQuadrupletTSST_Model = None

__all__ = [
    "ConvLSTM_Model",
    "EarthFarseer_Model",
    "HybridUNetFacTS",
    "LocalFRegionBranch",
    "LocalResidualRefiner",
    "MAU_Model",
    "MIM_Model",
    "PredFormerFacTS_Model",
    "PredFormerQuadrupletTSST_Model",
    "PredRNNpp_Model",
    "PredRNNv2_Model",
    "SUPPORTED_ARCHS",
    "SimVP",
    "SimVPForecast",
    "TAU_Model",
]
