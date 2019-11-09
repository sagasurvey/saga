import warnings

from astropy.table import TableReplaceWarning

warnings.filterwarnings("ignore", "numpy.dtype size changed", RuntimeWarning)
warnings.filterwarnings("ignore", category=TableReplaceWarning)
