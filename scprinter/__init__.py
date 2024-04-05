from .modelWrapper import *

__version__ = "0.0.1a"
import sys

from . import chromvar, datasets, fetch, genome, motifs, peak
from . import plotting as pl
from . import preprocessing as pp
from . import seq
from . import tools as tl
from . import utils
from .io import PyPrinter, load_printer

# from . import HMM


sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["pp", "tl", "pl"]})
