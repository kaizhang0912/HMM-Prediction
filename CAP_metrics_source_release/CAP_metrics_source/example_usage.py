"""Notebook usage with the four existing state-label dictionaries.

Import this function into the notebook after HMM decoding. Dictionaries must
use consistent, documented state correspondence before group comparisons.
"""
from pathlib import Path
from cap_metrics import export_metrics


def export_four_datasets(pre_pt, post_pt, pre_hc, post_hc, output_dir="./results/cap_metrics", k_states=8, tr=2.0):
    datasets = {"PT_pre": pre_pt, "PT_post": post_pt, "HC_pre": pre_hc, "HC_post": post_hc}
    return {name: export_metrics(data, Path(output_dir) / name, k_states, tr)
            for name, data in datasets.items()}


# In the analysis notebook:
# results = export_four_datasets(
#     state_index_dict_pre_pt, state_index_dict_post_pt,
#     state_index_dict_pre_hc, state_index_dict_post_hc,
#     k_states=k_states, tr=2.0
# )
