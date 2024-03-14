from .modelWrapper import *
__version__ = "0.0.1a"
from .io import PyPrinter, load_printer
from . import plotting as pl
from . import preprocessing as pp
from . import tools as tl
from . import genome
from . import motifs
from . import datasets
from . import fetch
from . import utils
from . import peak
from . import seq
from . import chromvar
# from . import HMM

import sys
sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['pp', 'tl', 'pl']})