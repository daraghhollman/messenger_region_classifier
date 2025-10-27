import os
import pathlib
import sys

sys.path.insert(
    1, str(pathlib.Path(os.path.dirname(__file__)) / ".." / "data" / "autometa")
)

from autometa import make_Metakernel

make_Metakernel.make_Metakernel("MESSENGER", ".")
