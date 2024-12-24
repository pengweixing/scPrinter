__version__ = "1.0.0a"
import sys

from . import chromvar, datasets, dorc, genome, motifs, peak
from . import plotting as pl
from . import preprocessing as pp
from . import seq
from . import tools as tl
from . import utils
from .io import load_printer, scPrinter
from .seq import Models, interpretation

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["pp", "tl", "pl"]})
