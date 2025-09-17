import sys
import os

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions with specific aliases to avoid name conflicts
from .data_processing import (
    load_npy_file,
    save_to_json,
    load_from_json,
    get_features,
    dict_to_np,
    discretize_features,
    convert_signal_to_complex,
    reduce_example_dict,
    get_processed_data,
    save_processed_data,
    load_processed_data
)

from .visualization import (
    visualize_tsne, 
    save_figure_as_html,
    generate_distinct_colors,
    get_marker_symbols,
    get_3d_marker_symbols,
    plot_confusion_matrix,
    )

__all__ = [
    # From data_processing
    'load_npy_file',
    'get_radioml_label',
    'dict_to_np',
    'get_radioml_snr',
    'get_processed_data',
    'save_processed_data',
    'load_processed_data',

    # From t_sne
    'visualize_tsne',
    'save_figure_as_html',
    'generate_distinct_colors',
    'get_marker_symbols',
    'get_3d_marker_symbols',
    'plot_confusion_matrix',
]