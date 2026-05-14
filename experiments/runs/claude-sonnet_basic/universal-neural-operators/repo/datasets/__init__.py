from .burgers import BurgersDataset, generate_burgers_data
from .gray_scott import GrayScottDataset, generate_gray_scott_data
from .navier_stokes import NavierStokesDataset, generate_navier_stokes_data
from .heat_equation import HeatEquationDataset, generate_heat_data
from .reaction_diffusion import ReactionDiffusionDataset, generate_reaction_diffusion_data
from .advection import AdvectionDataset, generate_advection_data
from .pdebench import PDEBenchDataset

__all__ = [
    "BurgersDataset", "generate_burgers_data",
    "GrayScottDataset", "generate_gray_scott_data",
    "NavierStokesDataset", "generate_navier_stokes_data",
    "HeatEquationDataset", "generate_heat_data",
    "ReactionDiffusionDataset", "generate_reaction_diffusion_data",
    "AdvectionDataset", "generate_advection_data",
    "PDEBenchDataset",
]
