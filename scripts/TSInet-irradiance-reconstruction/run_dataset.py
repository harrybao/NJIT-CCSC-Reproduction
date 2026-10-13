"""Run TSInet default-model reconstruction for a given test dataset.

Works around a CLI bug in tsinet_reconstructor.process_args(): it assigns
`data_test_file_name` and `dataset_name` as LOCAL variables, so `-n NRLTSI2`
(or SATIRE-S) silently reconstructs the default TCTE file. We set the module
globals explicitly; nothing else is changed.
Usage (cwd = TSInet working copy): python run_dataset.py DATASET
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import tsinet_reconstructor as m  # noqa: E402

ds = sys.argv[1]
m.process_args({"dataset_name": ds, "model_dir": "default_model"})
m.dataset_name = ds
m.data_test_file_name = os.path.join("test_data", ds + "_TSI.csv")
m.result_file_name = os.path.join("results", "fixed_" + ds + "_result.csv")
m.reconstruct_tsinet()
