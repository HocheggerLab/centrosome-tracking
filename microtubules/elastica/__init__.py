from pint import UnitRegistry

from ._planar_ivp import PlanarElasticaIVP, PlanarElasticaIVPArtist
from ._imageplanar import ImagePlanarElastica
from ._planarminimize import PlanarImageMinimizerIVP

ureg = UnitRegistry()
_q = ureg.Quantity

__version__ = "0.0.1"
