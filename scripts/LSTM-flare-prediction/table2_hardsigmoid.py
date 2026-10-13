"""Root-cause probe for the TF1->TF2 gap: Keras 2.2.4 (used for the paper) built
LSTM layers with recurrent_activation='hard_sigmoid' by default; tf.keras 2.x uses
'sigmoid'. We re-run table2_experiment.py with the old default patched in.
Usage (cwd = LSTM working copy): python table2_hardsigmoid.py CLASS SEED OUT_DIR
"""
import os
import runpy
import sys

import keras.layers as KL

_LSTM = KL.LSTM


class LSTM224(_LSTM):  # Keras 2.2.4 default
    def __init__(self, *a, **k):
        k.setdefault("recurrent_activation", "hard_sigmoid")
        super().__init__(*a, **k)


KL.LSTM = LSTM224
sys.path.insert(0, os.getcwd())
import LSTM_Flare  # noqa: E402
LSTM_Flare.LSTM = LSTM224  # the module did `from keras.layers import *`
sys.argv = ["table2_experiment.py"] + sys.argv[1:]
runpy.run_path(os.path.join(os.path.dirname(__file__), "table2_experiment.py"), run_name="__main__")
