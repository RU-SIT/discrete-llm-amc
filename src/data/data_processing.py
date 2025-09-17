"""
Data Processing Module for Signal Feature Extraction and Prompt Generation

This module provides functions for loading signal data (e.g., from .npy files),
extracting various statistical features, scaling and discretizing these features,
and generating formatted prompts suitable for language models, potentially including
few-shot examples.

Core functionalities include:
- Loading data from .npy, JSON, and pickle files.
- Calculating statistical features (mean, variance, moments, k-stats, etc.).
- Scaling features using StandardScaler.
- Discretizing features using KBinsDiscretizer.
- Converting feature dictionaries into NumPy arrays and text representations.
- Generating multiple-choice options strings.
- Constructing complete prompts with instructions, optional context (few-shot examples),
  and a question based on signal features (either continuous or discretized).
- A main processing function `get_processed_data` that orchestrates these steps.

Adapting for Other Datasets:
-----------------------------
This module currently uses helper functions specifically designed for the RadioML dataset
(imported from `radioml`: `get_radioml_label`, `get_radioml_snr`, `radioml_example_maker`).
These functions handle the specific file naming conventions and directory structure of RadioML
to extract labels, SNRs, and organize example files.

To use this module with a different dataset, you will likely need to:

1.  **Implement Custom Metadata Extraction:** Create your own functions similar to
    `get_radioml_label` and `get_radioml_snr` that can parse labels and any relevant
    metadata (like SNR, if applicable) from your dataset's file paths or filenames.
2.  **Implement Custom Example Handling:** If using few-shot examples (`add_context=True`),
    create a function similar to `radioml_example_maker` (or modify the data loading
    logic in the `if __name__ == "__main__":` block) to correctly identify and group
    example file paths according to their class labels for your dataset structure.
3.  **Update Main Script:** Modify the `if __name__ == "__main__":` block (or your
    own script that calls `get_processed_data`) to use your custom functions for
    loading paths, extracting labels/metadata, and creating the `example_dict`.
4.  **Adjust Feature List:** The `feature_names` list can be customized based on the
    statistical properties relevant to your specific signal type and task.

The core functions like `get_features`, `discretize_features`, `get_scaled_info`,
`generate_prompt`, and `get_processed_data` are generally dataset-agnostic, provided
they receive the correctly formatted inputs (signal paths, labels, SNRs, example dictionary).
"""
# -*- coding: utf-8 -*-
# TO DO LIST:
# 5. Test the code to generate numerical prompts, numerical features, textual features, contextual prompts.
# 6. Use helper functions to use for different datasets
# 7. Move RadioML specific functions to a separate file

# %%
from math import e
import os
import json
import random
import pickle
from glob import glob

import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from typing import List, Tuple, Dict, Union, Any, Optional
from sklearn.preprocessing import KBinsDiscretizer

def load_npy_file(file_path: str) -> np.ndarray:
    """
    Loads data from a .npy file located at the specified path.

    Args:
        file_path (str): The file system path to the .npy file.

    Returns:
        np.ndarray: A NumPy array containing the data loaded from the file.
    """
    # Use the numpy load function to read the array from the file
    data = np.load(file_path)
    # Return the loaded numpy array
    return data

def save_to_json(data: dict, file_path: str) -> None:
    """
    Saves a dictionary to a JSON file at the specified path.

    Args:
        data (dict): The dictionary containing the data to be saved.
        file_path (str): The file system path where the JSON file will be saved.

    Returns:
        None
    """
    # Open the file specified by file_path in write mode ('w')
    # The 'with' statement ensures the file is properly closed even if errors occur
    with open(file_path, 'w') as f:
        # Use the json.dump function to write the dictionary 'data' to the file object 'f'
        # 'indent=2' formats the JSON output with an indentation of 2 spaces for readability
        json.dump(data, f, indent=2)

def load_from_json(file_path: str) -> Any:
    """
    Loads data from a JSON file located at the specified path.

    Args:
        file_path (str): The file system path to the JSON file.

    Returns:
        Any: The Python object (often a dictionary or list) represented by the JSON data.
    """
    # Open the file specified by file_path in read mode ('r')
    # The 'with' statement ensures the file is properly closed even if errors occur
    with open(file_path, 'r') as f:
        # Use the json.load function to parse the JSON data from the file object 'f'
        # and return the corresponding Python object
        return json.load(f)
    
def save_processed_data(data: Any, file_path: str) -> None:
    """
    Saves processed data to a file using pickle serialization.

    Args:
        data (Any): The Python object (e.g., dictionary, list) containing the processed data to be saved.
                    This object must be pickleable.
        file_path (str): The file system path where the pickle file will be saved.

    Returns:
        None
    """
    # Open the specified file path in binary write mode ('wb').
    # The 'with' statement ensures the file is properly closed afterwards.
    with open(file_path, 'wb') as f:
        # Use pickle.dump to serialize the 'data' object and write it to the file object 'f'.
        pickle.dump(data, f)

def load_processed_data(file_path: str) -> Any:
    """
    Loads processed data from a file using pickle deserialization.

    Args:
        file_path (str): The file system path to the pickle file to be loaded.

    Returns:
        Any: The Python object that was deserialized from the pickle file.
             The actual type depends on what was originally saved in the file.
    """
    # Open the specified file path in binary read mode ('rb').
    # The 'with' statement ensures the file is properly closed afterwards.
    with open(file_path, 'rb') as f:
        # Use pickle.load to deserialize the data from the file object 'f'.
        loaded_data = pickle.load(f)
    # Return the loaded Python object.
    return loaded_data

def get_features(signal_data: np.ndarray, feature_names: List[str], snr=None) -> Dict[str, Union[float, np.ndarray, int]]:
    """
    Calculates specified statistical features for the given signal data.

    This function computes various descriptive statistics, moments, and k-statistics
    based on the names provided in the `feature_names` list.

    Args:
        signal_data (np.ndarray): A NumPy array containing the signal data for which
                                  to calculate features. It's expected to be a 1D or 2D array
                                  where stats.describe can operate column-wise or on the flattened array.
        feature_names (List[str]): A list of strings specifying the names of the features
                                   to calculate. Supported features include:
                                   - 'nobs': Number of observations
                                   - 'min': Minimum value
                                   - 'max': Maximum value
                                   - 'mean': Mean value
                                   - 'variance': Variance
                                   - 'skewness': Skewness
                                   - 'kurtosis': Kurtosis (Fisher's definition)
                                   - 'moment_k': The k-th central moment (e.g., 'moment_2' for variance)
                                   - 'kstat_n': The n-th k-statistic (unbiased estimator of cumulant)
                                   - 'kstatvar_n': The variance of the n-th k-statistic

    Returns:
        Dict[str, Union[float, np.ndarray, int]]: A dictionary where keys are the feature names
                                                  and values are the calculated statistical values.
                                                  The type of the value depends on the statistic
                                                  (e.g., 'nobs' is int, others are often float or potentially np.ndarray
                                                  if stats.describe returns array-like results for multi-column input).
    """
    # Initialize an empty dictionary to store the calculated features
    stats_summary: Dict[str, Union[float, np.ndarray, int]] = {}
    # Initialize descriptive statistics result to None. It will be calculated only if needed
    # to avoid redundant computation if only moments or k-stats are requested.
    des_stats: Any = None # Using Any because the type from stats.describe can be complex

    # Iterate through the requested feature names
    for feature in feature_names:
        if feature == 'snr':
            if snr is not None:
                stats_summary['snr'] = snr
        elif feature in ('nobs', 'min', 'max', 'mean', 'variance', 'skewness', 'kurtosis'):
            # If descriptive stats haven't been calculated yet for this signal, do it now.
            # This avoids re-calculating for each feature that needs it.
            if des_stats is None:
                # Calculate descriptive statistics for the signal data
                des_stats = stats.describe(signal_data, axis=None)
            # Assign the appropriate statistic from the calculated results
            stats_summary[feature] = getattr(des_stats, feature)
        elif feature.startswith('moment_'):
            # Extract the moment order from the feature name string (e.g., 'moment_2' -> 2)
            moment_order = int(feature.split('_')[1])
            # Calculate the central moment of the specified order
            stats_summary[feature] = stats.moment(signal_data, moment=moment_order, axis=None)
        elif feature.startswith('kstat_'):
            # Extract the k-statistic order from the feature name string (e.g., 'kstat_2' -> 2)
            kstat_order = int(feature.split('_')[1])
            # Calculate the k-statistic of the specified order
            stats_summary[feature] = stats.kstat(signal_data, n=kstat_order)
        elif feature.startswith('kstatvar_'):
            # Extract the k-statistic variance order from the feature name string (e.g., 'kstatvar_2' -> 2)
            kstatvar_order = int(feature.split('_')[1])
            # Calculate the variance of the k-statistic of the specified order
            stats_summary[feature] = stats.kstatvar(signal_data, n=kstatvar_order)
        else:
            # If the feature name is not recognized, raise an error
            raise ValueError(f"Unknown feature: {feature}")

    # Return the dictionary containing the calculated features
    return stats_summary

def dict_to_np(signal_summary: Dict[str, Union[float, np.ndarray, int]], feature_names: List[str]) -> np.ndarray:
    """
    Converts a dictionary of signal features into a 1D NumPy array.

    This function iterates through the specified feature names, retrieves the corresponding
    values from the input dictionary, and flattens them into a single NumPy array.
    If a feature value is iterable (like a list or NumPy array, but not a string/bytes),
    its elements are appended individually. Otherwise, the single value is appended.

    Args:
        signal_summary (Dict[str, Union[float, np.ndarray, int]]): A dictionary where keys are
                                                                    feature names (strings) and
                                                                    values are the calculated
                                                                    feature values (numeric or array-like).
        feature_names (List[str]): A list of strings representing the keys in `signal_summary`
                                   to include in the output array, in the desired order.

    Returns:
        np.ndarray: A 1D NumPy array containing the flattened feature values from the
                    dictionary, ordered according to `feature_names`. The dtype is set to float.
    """
    # Initialize an empty list to store the flattened feature values
    flat_features = []
    # Iterate through the feature names provided in the specified order
    for feature in feature_names:
        # Get the value associated with the current feature name from the dictionary
        value = signal_summary[feature]
        # Check if the value is iterable (e.g., list, tuple, np.ndarray)
        # but exclude strings and bytes, which are iterable but should be treated as single values here.
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            # If it's an iterable, extend the list with its elements
            flat_features.extend(value)
        else:
            # If it's a single value (or string/bytes), append it directly
            flat_features.append(value)

    # Convert the list of flattened features into a NumPy array with dtype float
    return np.array(flat_features, dtype=float)

def discretize_features(
    data: Dict[str, Any], 
    n_bins: int = 10, 
    strategy: str = 'uniform', 
    discretizers: Optional[Dict[int, KBinsDiscretizer]] = None
) -> Dict[str, Any]:
    """
    Discretizes continuous features stored in a dictionary using KBinsDiscretizer.

    This function takes a dictionary containing dataset information, extracts the 
    continuous features (expected under the key 'stats'), and discretizes each 
    feature column independently. It can either use pre-fitted discretizers 
    or fit new ones. The discretized data and the discretizers used are added 
    back into the dictionary.

    Args:
        data (Dict[str, Any]): A dictionary containing the processed data. It must 
                               contain a key 'stats' whose value is a NumPy array 
                               (n_samples, n_features) of continuous features.
        n_bins (int, optional): The number of bins to produce. Defaults to 10.
        strategy (str, optional): The strategy used to define the widths of the bins.
                                  Options are 'uniform', 'quantile', 'kmeans'. 
                                  Defaults to 'uniform'.
        discretizers (Optional[Dict[int, KBinsDiscretizer]], optional): 
                                  A dictionary mapping feature indices (int) to 
                                  pre-fitted KBinsDiscretizer objects. If None, new 
                                  discretizers will be fitted for each feature. 
                                  Defaults to None.

    Returns:
        Dict[str, Any]: The updated input dictionary with two new keys:
                        - 'stats_discretized': A NumPy array (n_samples, n_features) 
                          containing the discretized feature values (as bin indices).
                        - 'discretizers': A dictionary mapping feature indices (int) 
                          to the KBinsDiscretizer object used for that feature 
                          (either the ones provided or the newly fitted ones).
    """
    # Retrieve the continuous feature data from the input dictionary
    continuous_data: np.ndarray = data['stats']
    
    # Initialize the dictionary to store discretizers if none is provided
    if discretizers is None:
        discretizers: Dict[int, KBinsDiscretizer] = {}
        
    # Initialize an array to store the discretized data, with the same shape as the input
    # Using integer type because KBinsDiscretizer with encode='ordinal' returns bin indices
    discretized_data = np.zeros_like(continuous_data, dtype=int)
    
    # Iterate through each feature column for independent discretization
    for feature_idx in range(continuous_data.shape[1]):
        # Extract the current feature column and reshape it to be 2D (n_samples, 1)
        # as required by scikit-learn transformers
        feature_values = continuous_data[:, feature_idx].reshape(-1, 1)
        
        # Check if a pre-fitted discretizer was provided for this feature index
        if feature_idx in discretizers:
            # Use the existing discretizer to transform the data
            discretizer = discretizers[feature_idx]
            discretized_feature = discretizer.transform(feature_values)
        else:
            # No pre-fitted discretizer provided, so create a new one
            discretizer = KBinsDiscretizer(
                n_bins=n_bins, 
                encode='ordinal',  # Output bin indices
                strategy=strategy,
                subsample=None, # Use all data for fitting kmeans if strategy='kmeans'
                random_state=None # Default random state for kmeans
            )
            # Fit the discretizer to the current feature's data and then transform it
            discretized_feature = discretizer.fit_transform(feature_values)
            # Store the newly fitted discretizer in the dictionary
            discretizers[feature_idx] = discretizer

        # Store the resulting discretized values (flattened) into the corresponding column
        # of the output array
        discretized_data[:, feature_idx] = discretized_feature.flatten()
        
    # Add the array of discretized features to the data dictionary
    data['stats_discretized'] = discretized_data
    # Add the dictionary of used/fitted discretizers to the data dictionary
    data['discretizers'] = discretizers
    
    # Return the modified data dictionary
    return data

def convert_signal_to_complex(signal: np.ndarray) -> np.ndarray:
    """
    Converts a real-valued signal represented as a 2D NumPy array
    (where column 0 is real and column 1 is imaginary) into a
    complex-valued 1D NumPy array.

    Args:
        signal (np.ndarray): A 2D NumPy array of shape (n_samples, 2),
                             where signal[:, 0] contains the real parts and
                             signal[:, 1] contains the imaginary parts.

    Returns:
        np.ndarray: A 1D NumPy array of complex numbers representing the signal.
    """
    # Extract the real part (first column) of the signal
    real_part = signal[:, 0]
    # Extract the imaginary part (second column) of the signal
    imaginary_part = signal[:, 1]
    # Combine the real and imaginary parts into a complex array
    # using the formula: complex_number = real + j * imaginary
    complex_signal = real_part + 1j * imaginary_part
    # Return the resulting 1D complex NumPy array
    return complex_signal

def split_real_imaginary(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits a complex-valued signal represented as a 1D NumPy array
    into its real and imaginary components.

    Args:
        signal (np.ndarray): A 1D NumPy array of complex numbers.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two 1D NumPy arrays:
            - The first array contains the real parts of the complex numbers.
            - The second array contains the imaginary parts of the complex numbers.
    """
    # Separate the real and imaginary parts using NumPy's built-in functions
    real_part = np.real(signal)
    imaginary_part = np.imag(signal)
    signal = np.column_stack((real_part, imaginary_part))
    return signal

def reduce_example_dict(example_dict: Dict[str, Any], label: str, max_examples: int = 10) -> Dict[str, Any]:
    """
    Reduces the number of examples in a dictionary, ensuring the example with the specified label is kept.

    If the number of keys (examples) in the input dictionary exceeds `max_examples`,
    this function filters the dictionary to include the example corresponding to the
    given `label` and a random sample of `max_examples - 1` other examples.
    The resulting dictionary is then shuffled. If the number of examples is already
    less than or equal to `max_examples`, the original dictionary is returned unchanged.

    Args:
        example_dict (Dict[str, Any]): The dictionary containing examples, where keys
                                       are typically labels or identifiers.
        label (str): The specific label (key) that must be included in the reduced dictionary.
        max_examples (int, optional): The maximum number of examples allowed in the
                                      returned dictionary. Defaults to 10.

    Returns:
        Dict[str, Any]: A dictionary containing at most `max_examples` items, including
                        the item associated with `label`, and potentially shuffled.
                        Returns the original dictionary if its size is within the limit.
    """
    # Get the list of keys (labels/identifiers) from the input dictionary
    example_dict_keys: List[str] = list(example_dict.keys())

    # Check if the number of examples exceeds the specified maximum
    if len(example_dict_keys) > max_examples:
        # Filter keys to separate those not matching the target label
        filtered_keys: List[str] = [key for key in example_dict_keys if label not in key]
        # Filter keys to get the one(s) matching the target label
        # Assumes there's at least one key containing the label
        label_keys: List[str] = [key for key in example_dict_keys if label in key]
        
        # Randomly sample 'max_examples - 1' keys from the filtered (non-label) keys
        # This ensures space for the target label key
        sample_keys: List[str] = random.sample(filtered_keys, max_examples - 1)

        # Create a new dictionary containing the randomly sampled examples
        new_example_dict: Dict[str, Any] = {k: example_dict[k] for k in sample_keys}

        # Add the example corresponding to the target label to the new dictionary
        # Uses the first key found that contains the label
        new_example_dict[label_keys[0]] = example_dict[label_keys[0]]

        # Shuffle the items (key-value pairs) in the new dictionary randomly
        items: List[tuple[str, Any]] = list(new_example_dict.items())
        random.shuffle(items)
        # Convert the shuffled list of items back into a dictionary
        new_example_dict = dict(items)
    else:
        # If the number of examples is within the limit, return the original dictionary
        new_example_dict = example_dict

    # Return the potentially reduced and shuffled dictionary
    return new_example_dict

def create_options(class_names: List[str]) -> str:
    """
    Generates a formatted string representing multiple-choice options from a list of class names.

    The format is "[A: class_name1, B: class_name2, ..., Z: class_nameN]".
    It supports up to 26 options (A-Z).

    Args:
        class_names (List[str]): A list of strings, where each string is a class name
                                 to be included as an option.

    Returns:
        str: A formatted string representing the options.

    Raises:
        AssertionError: If the number of class names is less than 1 or greater than 26.
    """
    # Ensure the number of class names is within the valid range (1 to 26 for A-Z)
    assert 1 <= len(class_names) <= 26, "Class names must be between 1 and 26 to map to A-Z."

    # Start building the options string with an opening bracket
    options_str = "["
    # Iterate through the class names with their indices (0-based)
    for i, class_name in enumerate(class_names):
        # Calculate the ASCII value for the corresponding uppercase letter (A=65, B=66, ...)
        # chr(64 + i + 1) converts the index (0, 1, 2...) to the character ('A', 'B', 'C'...)
        # Append the formatted option (e.g., "A: class_name, ") to the string
        options_str += f"{chr(64 + i + 1)}: {class_name}, "

    # Remove the trailing comma and space added by the last iteration
    options_str = options_str[:-2]
    # Add the closing bracket to complete the string
    options_str += "]"
    # Return the fully formatted options string
    return options_str

def discretize_features(
    signal_stats: np.ndarray, 
    n_bins: int = 10, 
    strategy: str = 'uniform', 
    discretizers: Optional[Dict[int, KBinsDiscretizer]] = None
) -> Tuple[np.ndarray, Dict[int, KBinsDiscretizer]]:
    """
    Discretizes each feature (column) in the provided numerical data independently.

    This function applies binning to each column of the input NumPy array. 
    It can either use pre-fitted KBinsDiscretizer objects provided via the 
    `discretizers` dictionary or fit new ones if they are not provided for a 
    specific feature index.

    Args:
        signal_stats (np.ndarray): A 2D NumPy array where rows represent samples 
                                   and columns represent features (n_samples, n_features).
        n_bins (int, optional): The number of bins to produce for discretization. 
                                Defaults to 10.
        strategy (str, optional): The strategy used to define the widths of the bins.
                                  Options: 'uniform', 'quantile', 'kmeans'. 
                                  Defaults to 'uniform'.
        discretizers (Optional[Dict[int, KBinsDiscretizer]], optional): 
                                  A dictionary mapping feature indices (int) to 
                                  pre-fitted KBinsDiscretizer objects. If None or if 
                                  a feature index is missing, a new discretizer will 
                                  be fitted for that feature. Defaults to None.

    Returns:
        Tuple[np.ndarray, Dict[int, KBinsDiscretizer]]: A tuple containing:
            - discretized_data (np.ndarray): A NumPy array of the same shape as 
              `signal_stats` but with discretized values (bin indices), dtype=int.
            - discretizers (Dict[int, KBinsDiscretizer]): The dictionary mapping 
              feature indices to the KBinsDiscretizer objects used (either the 
              ones provided or the newly fitted ones).
    """
    # Ensure the input is a NumPy array for consistent processing
    continuous_data: np.ndarray = np.array(signal_stats)
    
    # Initialize the dictionary to store discretizers if none is provided
    if discretizers is None:
        discretizers: Dict[int, KBinsDiscretizer] = {}
        
    # Initialize an array to store the discretized data, matching the input shape
    # Using integer type because KBinsDiscretizer with encode='ordinal' returns bin indices
    discretized_data = np.zeros_like(continuous_data, dtype=int)
    
    # Iterate through each feature column (index) for independent discretization
    for feature_idx in range(continuous_data.shape[1]):
        # Extract the current feature column (all samples for this feature)
        # Reshape to (-1, 1) -> a 2D array with one column, as required by scikit-learn transformers
        feature_values: np.ndarray = continuous_data[:, feature_idx].reshape(-1, 1)
        
        # Check if a pre-fitted discretizer was provided for this specific feature index
        if feature_idx in discretizers:
            # Use the existing discretizer from the dictionary
            discretizer: KBinsDiscretizer = discretizers[feature_idx]
            # Transform the feature values using the pre-fitted discretizer
            discretized_feature: np.ndarray = discretizer.transform(feature_values)
        else:
            # No pre-fitted discretizer found for this index, so create and fit a new one
            discretizer = KBinsDiscretizer(
                n_bins=n_bins, 
                encode='ordinal',  # Output bin indices (0 to n_bins-1)
                strategy=strategy,
                # Add necessary parameters if using 'kmeans' like subsample and random_state if needed
                # subsample=None, 
                # random_state=None 
            )
            # Fit the new discretizer to the current feature's data and then transform the data
            discretized_feature = discretizer.fit_transform(feature_values)
            # Store the newly fitted discretizer in the dictionary for potential reuse or inspection
            discretizers[feature_idx] = discretizer
        
        # Store the resulting discretized values (bin indices) into the corresponding column
        # of the output array. Flatten the result to ensure it fits into the 1D slice.
        discretized_data[:, feature_idx] = discretized_feature.flatten()

    # Return the array containing the discretized data and the dictionary of discretizers used
    return discretized_data, discretizers

def get_feature_dim(info: Dict[str, Any]) -> Dict[str, int]:
    """
    Calculates the dimension (number of elements) for each feature in an info dictionary.

    Iterates through the values in the input dictionary. If a value is iterable 
    (like a list or NumPy array, excluding strings/bytes), its length is used as 
    the dimension. Otherwise (for single values like numbers), the dimension is 1.

    Args:
        info (Dict[str, Any]): A dictionary where keys are feature names (str) 
                               and values are the feature values (can be single 
                               numbers, lists, arrays, etc.).

    Returns:
        Dict[str, int]: A dictionary where keys are the same feature names (str) 
                        and values are the calculated dimensions (int) for each feature.
    """
    # Initialize an empty dictionary to store the dimension of each feature
    feature_dim: Dict[str, int] = {}
    # Iterate through each key (feature name) in the input dictionary
    for key in info.keys():
        # Check if the value associated with the key is iterable (e.g., list, tuple, np.ndarray)
        # Exclude strings and bytes, which are iterable but usually represent single entities here.
        if hasattr(info[key], '__iter__') and not isinstance(info[key], (str, bytes)):
            # If it's an iterable, its dimension is its length
            feature_dim[key] = len(info[key])
        else:
            # If it's not an iterable (or it's a string/bytes), its dimension is 1
            feature_dim[key] = 1
    # Return the dictionary containing feature names and their corresponding dimensions
    return feature_dim

def get_discrete_info(
    info: Dict[str, Union[float, np.ndarray, int]], 
    discretizers: Dict[int, KBinsDiscretizer]
) -> Dict[str, Union[int, np.ndarray]]:
    """
    Discretizes feature values stored in a dictionary using pre-fitted discretizers.

    This function takes a dictionary containing feature names and their corresponding 
    numerical values (which might include iterables like min/max pairs). It first 
    flattens these values into a 1D NumPy array. Then, using a provided dictionary 
    of pre-fitted KBinsDiscretizer objects (keyed by the index in the flattened array), 
    it transforms each flattened value into its corresponding discrete bin index. 
    Finally, it reconstructs a dictionary with the same keys as the input `info` 
    dictionary, but with the discretized integer values (or arrays of integers for 
    original iterables).

    Args:
        info (Dict[str, Union[float, np.ndarray, int]]): 
            A dictionary where keys are feature names (str) and values are the 
            calculated feature values (numeric or array-like, e.g., for min/max).
        discretizers (Dict[int, KBinsDiscretizer]): 
            A dictionary mapping the index of a feature in the flattened representation 
            (obtained via `dict_to_np`) to its corresponding pre-fitted 
            KBinsDiscretizer object.

    Returns:
        Dict[str, Union[int, np.ndarray]]: 
            A dictionary with the same keys as the input `info` dictionary, but 
            where the values have been replaced by their discretized bin indices (int). 
            If the original value was an iterable, the corresponding value in the 
            output dictionary will be a NumPy array of integer bin indices.
    """
    # Convert the input dictionary of features into a 1D NumPy array.
    # Note: This flattens any iterable values in the dictionary.
    np_info: np.ndarray = dict_to_np(info, list(info.keys())) 
    
    # Initialize a NumPy array to store the discretized results, matching the shape and using integer type.
    discretized_summary = np.zeros_like(np_info, dtype=int)

    # Iterate through each element (feature value) in the flattened NumPy array.
    for feature_idx in range(np_info.shape[0]):
        # Extract the single feature value at the current index.
        # Reshape it to a 2D array (1 sample, 1 feature) as required by scikit-learn transformers.
        feature_values = np_info[feature_idx].reshape(-1, 1)
        
        # Retrieve the appropriate pre-fitted discretizer for this feature index.
        # Assumes a discretizer exists for every index in the flattened array.
        discretizer: KBinsDiscretizer = discretizers[feature_idx]
        
        # Transform the feature value using the discretizer to get the bin index.
        discretized_feature: np.ndarray = discretizer.transform(feature_values)
        
        # Store the resulting bin index (as an integer) in the summary array.
        # .ravel()[0] extracts the single integer value from the (1, 1) output array.
        discretized_summary[feature_idx] = discretized_feature.ravel()[0]

    # Initialize an empty dictionary to store the reconstructed discretized information.
    signal_info: Dict[str, Union[int, np.ndarray]] = {}
    # Initialize an index counter to track the position in the `discretized_summary` array.
    i = 0
    # Iterate through the keys of the original `info` dictionary to maintain structure.
    for key in info.keys():
        # Check if the original value associated with this key was iterable (and not str/bytes).
        if hasattr(info[key], '__iter__') and not isinstance(info[key], (str, bytes)):
            # If it was iterable, assume it corresponded to two consecutive values 
            # in the flattened array (e.g., min and max). Reconstruct as a NumPy array.
            # Note: This assumes original iterables always had length 2.
            signal_info[key] =  np.array([discretized_summary[i], discretized_summary[i+1]], dtype=int)
            # Increment the counter by 2.
            i += 2
        else:
            # If the original value was a scalar, take the single corresponding value 
            # from the discretized summary.
            signal_info[key] = discretized_summary[i]
            # Increment the counter by 1.
            i += 1
    
    # Return the reconstructed dictionary containing discretized feature values.
    return signal_info

def get_scaled_info(
    info: Dict[str, Union[float, np.ndarray, int]], 
    scaler: StandardScaler # Or a more general type like BaseEstimator if other scalers are used
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Scales feature values stored in a dictionary using a pre-fitted scaler.

    This function takes a dictionary of feature names and their numerical values 
    (which might include iterables like min/max pairs). It first converts these 
    values into a 1D NumPy array, reshapes it for the scaler, and then applies 
    the provided pre-fitted scaler (e.g., StandardScaler) to transform the values. 
    Finally, it reconstructs a dictionary with the same keys as the input `info` 
    dictionary, but with the scaled floating-point values (or NumPy arrays of 
    floats for original iterables).

    Args:
        info (Dict[str, Union[float, np.ndarray, int]]): 
            A dictionary where keys are feature names (str) and values are the 
            calculated feature values (numeric or array-like).
        scaler (StandardScaler): 
            A pre-fitted scikit-learn scaler object (e.g., StandardScaler) 
            used to transform the feature values.

    Returns:
        Dict[str, Union[float, np.ndarray]]: 
            A dictionary with the same keys as the input `info` dictionary, but 
            where the values have been replaced by their scaled versions (float). 
            If the original value was an iterable, the corresponding value in the 
            output dictionary will be a NumPy array of float scaled values.
    """
    # Convert the input dictionary of features into a 1D NumPy array using the keys for order.
    # Reshape the array to (1, -1) because scalers expect a 2D array (n_samples, n_features).
    np_info: np.ndarray = dict_to_np(info, list(info.keys())).reshape(1, -1)
    
    # Apply the pre-fitted scaler to transform the feature values.
    # The result `scaled_summary` will be a 2D NumPy array (1, n_features).
    scaled_summary: np.ndarray = scaler.transform(np_info)
    
    # Initialize an empty dictionary to store the reconstructed scaled information.
    signal_info: Dict[str, Union[float, np.ndarray]] = {}
    # Initialize an index counter to track the position in the `scaled_summary` array.
    i = 0
    # Iterate through the keys of the original `info` dictionary to maintain structure.
    for key in info.keys():
        # Check if the original value associated with this key was iterable (and not str/bytes).
        if hasattr(info[key], '__iter__') and not isinstance(info[key], (str, bytes)):
            # If it was iterable, assume it corresponded to two consecutive values 
            # in the flattened scaled array (e.g., scaled min and max). 
            # Reconstruct as a NumPy array of floats.
            # Note: This assumes original iterables always had length 2.
            signal_info[key] =  np.array([scaled_summary[0, i], scaled_summary[0, i+1]])
            # Increment the counter by 2.
            i += 2
        else:
            # If the original value was a scalar, take the single corresponding scaled value 
            # from the `scaled_summary` array. scaled_summary[0, i] accesses the value.
            signal_info[key] = scaled_summary[0, i]
            # Increment the counter by 1.
            i += 1
    
    # Return the reconstructed dictionary containing scaled feature values.
    return signal_info

def get_text_info(info: Dict[str, Union[float, int, np.ndarray]], decimal_precision: int = 3) -> str:
    """
    Converts a dictionary of feature information into a formatted text string.

    Iterates through the dictionary items. If a value is iterable (and not a string/bytes), 
    it formats it as "key: (value1, value2, ...)", rounding each number. 
    Otherwise, it formats it as "key: value, ", rounding the number.

    Args:
        info (Dict[str, Union[float, int, np.ndarray]]): 
            A dictionary where keys are feature names (str) and values are 
            numerical data (float, int, or NumPy array/list of numbers).
        decimal_precision (int, optional): 
            The number of decimal places to round the numerical values to. 
            Defaults to 3.

    Returns:
        str: A formatted string representation of the dictionary content, with 
             numbers rounded and a trailing comma and space after each entry.
    """
    # Initialize an empty string to build the result
    text_info = ""
    # Iterate through each feature name (key) in the input dictionary
    for key in info.keys():
        # Get the value associated with the current key
        value = info[key]
        # Check if the value is iterable (like list, tuple, np.ndarray) 
        # but not a string or bytes object
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            # If iterable, format as "key: (rounded_val1, rounded_val2, ...), "
            # Use a list comprehension to round each element in the iterable
            rounded_values = [f"{round(x, ndigits=decimal_precision)}" for x in value]
            # Join the rounded values with ", " and enclose in parentheses
            text_info += f"{key}: (" + ', '.join(rounded_values) + "), "
        else:
            # If not iterable (or string/bytes), format as "key: rounded_value, "
            # Round the single numerical value to the specified precision
            text_info += f"{key}: {round(value, ndigits=decimal_precision)}, "
            
    # Return the constructed string containing all formatted key-value pairs
    return text_info

def _to_base26_string(n: int) -> str:
    """Converts a 0-indexed integer to a base-26 string (A, B,..., Z, AA, AB...)."""
    if n < 0:
        return ""
    result = ""
    num = n + 1
    while num > 0:
        rem = (num - 1) % 26
        result = chr(65 + rem) + result
        num = (num - 1) // 26
    return result

def get_discrete_text_info(info: Dict[str, Union[int, np.ndarray]]) -> str:
    """
    Converts a dictionary of discretized feature information (bin indices) into a formatted text string,
    mapping integer bin indices to corresponding uppercase letters (0->A, 1->B, etc.).

    Iterates through the dictionary items. If a value is iterable (e.g., a NumPy array of bin indices),
    it formats it as "key: (Letter1, Letter2, ...)", converting each index to a letter.
    Otherwise (for single integer bin indices), it formats it as "key: Letter, ".

    Args:
        info (Dict[str, Union[int, np.ndarray]]):
            A dictionary where keys are feature names (str) and values are
            discretized feature values, represented as integer bin indices
            (int or NumPy array of ints). Assumes bin indices start from 0.

    Returns:
        str: A formatted string representation of the discretized dictionary content,
             with bin indices converted to letters (A=0, B=1, ...) and a trailing
             comma and space after each entry.
    """
    # Initialize an empty string to build the result
    text_info = ""
    # Iterate through each feature name (key) in the input dictionary
    for key in info.keys():
        # Get the value (discretized bin index or array of indices) associated with the current key
        value = info[key]
        # Check if the value is iterable (like a NumPy array containing multiple bin indices)
        # Exclude strings and bytes, although they are unlikely given the type hint.
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            # If iterable, format as "key: (Letter1, Letter2, ...), "
            # Convert each integer bin index `x` to its corresponding letter string (A, AA, etc.)
            letter_values = [_to_base26_string(int(x)) for x in value]
            # Join the resulting letters with ", " and enclose in parentheses
            text_info += f"{key}: (" + ', '.join(letter_values) + "), "
        else:
            # If not iterable (it's a single integer bin index), format as "key: Letter, "
            # Convert the single integer bin index to its corresponding letter string
            text_info += f"{key}: {_to_base26_string(int(value))}, "


    # Return the constructed string containing all formatted key-value pairs
    return text_info

def get_question_answer(
    signal: Union[np.ndarray, Dict[str, Union[float, int, np.ndarray]]], 
    options: List[str], 
    template: str,
    template_format: List[str], 
    answer: Optional[str] = None, 
    feature_names: Optional[List[str]] = None, 
    processed: bool = True,
    snr: Optional[float] = None,
    decimal_precision: Optional[int] = None, 
    discretizers: Optional[Dict[int, KBinsDiscretizer]] = None, 
    scaler: Optional[StandardScaler] = None, 
    discretized: bool = False
) -> str:
    """
    Generates a question or a question-answer pair based on signal features.

    This function takes either raw signal data or pre-processed features, 
    formats the features into a text string (either continuous rounded values 
    or discrete letter codes), and inserts this information, along with 
    multiple-choice options and optionally the correct answer letter, into a 
    provided template string.

    Args:
        signal (Union[np.ndarray, Dict[str, Union[float, int, np.ndarray]]]): 
            Either the raw signal data (if processed=False) or a dictionary 
            containing pre-calculated features (if processed=True).
        options (List[str]): A list of possible class names (answers) for the signal.
        template (str): A format string with placeholders for signal information, 
                        options, and the answer letter (e.g., "Info: {}\nOptions: {}\nAnswer: {}").
        template_format (List[str]): A list of format specifiers for the template string.
                                      Each specifier corresponds to a placeholder in the template.
        answer (Optional[str], optional): The correct class name for the signal. 
                                          If provided, its corresponding letter (A, B, C...) 
                                          will be included in the output. Defaults to None.
        feature_names (Optional[List[str]], optional): A list of feature names to 
                                                      calculate if processed=False. 
                                                      Required if processed=False. Defaults to None.
        processed (bool, optional): Flag indicating if the input `signal` is already 
                                    a dictionary of features (True) or raw data (False). 
                                    Defaults to True.
        decimal_precision (Optional[int], optional): The number of decimal places for 
                                                     formatting continuous features. 
                                                     Required if discretized=False. Defaults to None.
        discretizers (Optional[Dict[int, KBinsDiscretizer]], optional): 
                                                     A dictionary mapping feature indices to 
                                                     pre-fitted discretizers. Required if 
                                                     discretized=True and processed=False. 
                                                     Defaults to None.
        scaler (Optional[StandardScaler], optional): A pre-fitted scaler object. Required if 
                                                     discretized=False and processed=False. 
                                                     Defaults to None.
        discretized (bool, optional): Flag indicating whether to use discretized features 
                                      (True) or scaled continuous features (False). 
                                      Defaults to False.

    Returns:
        str: The formatted text string based on the template, containing the signal 
             information, options, and optionally the answer letter.
             
    Raises:
        AssertionError: If required arguments (feature_names, discretizers, scaler, 
                        decimal_precision) are missing based on the `processed` and 
                        `discretized` flags.
    """
    # Determine the initial signal information: use directly if processed, else calculate features.
    signal_info: Dict[str, Union[float, int, np.ndarray]] = signal if processed else get_features(signal, feature_names, snr=snr) # type: ignore

    # Process based on whether discretized output is requested.
    if discretized:
        # Ensure discretizers are provided if processing raw data.
        if not processed:  
            assert (discretizers is not None), "Discretizers must be provided when processing raw data for discretized output."
        # Get discrete feature values (bin indices) if not already processed, or use existing ones.
        # Note: If 'signal' was already processed discrete features, this step is skipped if processed=True.
        # If 'signal' was processed continuous features, this needs adjustment or assumes 'signal' contains discrete info.
        # Assuming 'signal' contains the correct type based on 'processed' flag for this branch.
        discrete_signal_info: Dict[str, Union[int, np.ndarray]] = signal_info if processed else get_discrete_info(signal_info, discretizers) # type: ignore
        # Convert the discrete bin indices into a text string with letter codes (A, B, ...).
        formatted_signal_info: str = get_discrete_text_info(discrete_signal_info)
    else:
        # Ensure decimal precision is provided for continuous output.
        assert (isinstance(decimal_precision, int)) and (decimal_precision > 0), "Decimal precision must be a positive integer for continuous output."
        # Ensure a scaler is provided if processing raw data.
        if not processed:
            assert (scaler is not None), "Scaler must be provided when processing raw data for continuous output."
        # Get scaled feature values if not already processed, or use existing ones.
        # Note: Similar logic as above applies regarding the type of 'signal_info' if processed=True.
        scaled_signal_info: Dict[str, Union[float, np.ndarray]] = signal_info if processed else get_scaled_info(signal_info, scaler) # type: ignore
        # Convert the scaled continuous values into a formatted text string with specified precision.
        formatted_signal_info: str = get_text_info(scaled_signal_info, decimal_precision)
    
    # Determine the answer letter (A, B, ...) if an answer is provided.
    # answer_letter: str = chr(64 + options.index(answer) + 1) if answer else ""
    answer_letter= answer if answer else ""
    
    # Format the final text using the provided template.
    text: str = template.format(formatted_signal_info, *template_format, answer_letter)
    
    # Return the completed question/answer string.
    return text

def ktop_example(k_top: List[str], example_dict: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
    """
    Selects the top-k examples from the example dictionary based on the provided keys.

    Args:
        k_top (List[str]): List of keys to select from the example dictionary.
        example_dict (Dict[str, List[np.ndarray]]): Dictionary containing example data.

    Returns:
        Dict[str, List[np.ndarray]]: A new dictionary containing only the selected top-k examples.
    """
    return {k: example_dict[k] for k in k_top if k in example_dict}

def generate_prompt(
    signal_data: Union[np.ndarray, Dict[str, Union[float, int, np.ndarray]]], 
    question_template: str,
    question_template_format: List[str], 
    instruction_template: str, 
    instruction_template_format: List[str],
    feature_names: List[str], 
    options: List[str],
    processed: bool = True, 
    add_context: bool = True, 
    example_dict: Optional[Dict[str, List[np.ndarray]]] = None, 
    decimal_precision: int = 3, 
    discretizers: Optional[Dict[int, KBinsDiscretizer]] = None, 
    scaler: Optional[StandardScaler] = None, 
    discretized: bool = False, 
    example_per_class: int = 1
) -> str:
    """
    Generates a complete prompt string including instructions, optional few-shot examples (context), 
    and the final question based on signal data or features.

    This function orchestrates the creation of a prompt suitable for a language model. 
    It first generates the question part using `get_question_answer`. If `add_context` 
    is True, it then constructs few-shot examples using the provided `example_dict`, 
    formatting each example as a question-answer pair using `get_question_answer`. 
    Finally, it combines the instruction template (filled with the context) and the 
    generated question.

    Args:
        signal_data (Union[np.ndarray, Dict[str, Union[float, int, np.ndarray]]]): 
            The input signal data. Can be raw signal (np.ndarray if processed=False) 
            or a dictionary of pre-calculated features (if processed=True).
        question_template (str): 
            The template string for formatting the question part (e.g., "Info: {}\nOptions: {}\nAnswer: {}").
        question_template_format (List[str]):
            A list of format specifiers for the question template, used to insert
        instruction_template (str): 
            The template string for the overall instruction, which should include a 
            placeholder for the context (e.g., "Based on the examples:\n{}\nAnswer the following question:").
        instruction_template_format (List[str]):
            A list of format specifiers for the instruction template, used to insert
        feature_names (List[str]): 
            List of feature names to be used, especially if `processed` is False.
        options (List[str]): 
            A list of possible class names (answers) for the signal, used for formatting options.
        processed (bool, optional): 
            Flag indicating if `signal_data` is already processed features (True) or raw data (False). 
            Defaults to True.
        add_context (bool, optional): 
            Flag indicating whether to add few-shot examples (context) to the prompt. 
            Defaults to True.
        example_dict (Optional[Dict[str, List[np.ndarray]]], optional): 
            A dictionary where keys are class labels (str) and values are lists of raw signal 
            examples (np.ndarray) for that class. Required if `add_context` is True. 
            Defaults to None.
        decimal_precision (int, optional): 
            Number of decimal places for formatting continuous features. Used by `get_question_answer`. 
            Defaults to 3.
        discretizers (Optional[Dict[int, KBinsDiscretizer]], optional): 
            Pre-fitted discretizers, used if `discretized` is True. Passed to `get_question_answer`. 
            Defaults to None.
        scaler (Optional[StandardScaler], optional): 
            Pre-fitted scaler, used if `discretized` is False. Passed to `get_question_answer`. 
            Defaults to None.
        discretized (bool, optional): 
            Flag indicating whether to use discretized features (True) or scaled continuous features (False). 
            Passed to `get_question_answer`. Defaults to False.
        example_per_class (int, optional): 
            The number of examples to include in the context for each class. 
            Defaults to 1.

    Returns:
        str: The fully constructed prompt string, including instructions, context (if added), and the question.
        
    Raises:
        AssertionError: If `add_context` is True but `example_dict` is None.
    """
    # Generate the question part of the prompt using the provided signal data/features.
    # The answer is set to None as this is the part the model needs to predict.
    question: str = get_question_answer(
        signal=signal_data, 
        options=options, 
        template=question_template, 
        template_format=question_template_format,
        answer=None,  # No answer provided for the main question
        processed=processed, 
        feature_names=feature_names, 
        decimal_precision=decimal_precision, 
        discretizers=discretizers, 
        scaler=scaler, 
        discretized=discretized
    )
    
    # Initialize context string
    context: str = ""
    # If context (few-shot examples) is requested:
    if add_context:
        # Ensure that the example dictionary is provided
        assert example_dict is not None, "Example dictionary must be provided for context."
        # Iterate through each class label in the example dictionary
        for key in example_dict.keys():
            # Include a specified number of examples per class
            for i in range(example_per_class):
                # Check if enough examples exist for the current class
                if i < len(example_dict[key]):
                    # Generate a question-answer pair for the example signal.
                    # Note: processed=False because examples in example_dict are raw signals.
                    # The correct answer (key) is provided for the example.
                    context += get_question_answer(
                        signal=example_dict[key][i][0] if isinstance(example_dict[key][i], tuple) else example_dict[key][i],  # Access the signal part of the tuple
                        options=options, 
                        template=question_template,
                        template_format=question_template_format,
                        answer=key, # Provide the correct label for the example
                        processed=False, # Examples are raw signals
                        snr=example_dict[key][i][1] if (isinstance(example_dict[key][i], tuple) and len(example_dict[key][i]) > 1) else None, # Access SNR if available
                        feature_names=feature_names, 
                        decimal_precision=decimal_precision, 
                        discretizers=discretizers, 
                        scaler=scaler, 
                        discretized=discretized
                    ) + "\n" # Add a newline after each example
            
    # Format the instruction template, inserting the generated context (few-shot examples)
    instruct: str = instruction_template.format(*instruction_template_format, context)
    
    # Combine the formatted instruction (with context) and the main question
    prompt: str = instruct + question

    # Return the final prompt string
    return prompt
    
def get_processed_data(
    signal_paths: List[str], 
    signal_labels: List[str], 
    signal_snr: List[Union[int, float]], 
    feature_names: Optional[List[str]], 
    question_template: str, 
    instruction_template: str, 
    example_dict: Optional[Dict[str, List[np.ndarray]]], 
    scaler: Optional[StandardScaler] = None,
    discretizers: Optional[Dict[int, KBinsDiscretizer]] = None,
    decimal_precision: int = 3, 
    add_context: bool = True
) -> Dict[str, Any]:
    """
    Processes signal data from file paths to generate features, prompts, and metadata.

    This function loads signals, calculates statistical features, scales and discretizes 
    these features, and generates textual prompts (both continuous and discretized) 
    with optional few-shot context.

    Args:
        signal_paths (List[str]): List of file paths to the signal data (.npy files).
        signal_labels (List[str]): List of corresponding labels for each signal.
        signal_snr (List[Union[int, float]]): List of corresponding Signal-to-Noise Ratios.
        feature_names (Optional[List[str]]): List of feature names to calculate. 
                                             If None, a default list is used.
        question_template (str): Template string for generating questions.
        instruction_template (str): Template string for generating instructions (including context).
        example_dict (Optional[Dict[str, List[np.ndarray]]]): Dictionary mapping labels to lists 
                                                               of example signals for few-shot context. 
                                                               Required if add_context is True.
        scaler (Optional[StandardScaler], optional): Pre-fitted StandardScaler. If None, a new one 
                                                     is fitted to the data. Defaults to None.
        discretizers (Optional[Dict[int, KBinsDiscretizer]], optional):
            Dictionary mapping feature indices to pre-fitted KBinsDiscretizer objects.
            If None, new discretizers are created and fitted to the data. Defaults to None.
        decimal_precision (int, optional): Number of decimal places for formatting continuous features 
                                           in prompts. Defaults to 3.
        add_context (bool, optional): Whether to add few-shot examples to the prompts. Defaults to True.

    Returns:
        Dict[str, Any]: A dictionary containing the processed data:
            - 'signals': List of loaded signal data (np.ndarray).
            - 'stats': List of dictionaries containing scaled features for each signal.
            - 'discret_stats': List of dictionaries containing discretized features (bin indices).
            - 'labels': Original list of signal labels.
            - 'snrs': Original list of signal SNRs.
            - 'prompts': List of generated prompts using scaled continuous features.
            - 'discret_prompts': List of generated prompts using discretized features.
            - 'feature_names': List of feature names used.
            - 'scaler': The StandardScaler object used (fitted or provided).
            - 'discretizers': Dictionary of KBinsDiscretizer objects used for each feature index.
            - 'num_samples': Total number of signals processed.
            - 'num_features': Number of features extracted per signal (after flattening).
            - '#classes': Number of unique classes found in labels.
            - '#snr': Number of unique SNR values found.
    """
    # Define default feature names if none are provided
    if not feature_names:
        feature_names = ['nobs', 'min', 'max', 'mean', 'variance', 'skewness', 'kurtosis', 
                         'moment_0', 'moment_1', 'moment_2', 'moment_3', 'moment_4', 
                         'moment_5', 'moment_6', 'moment_7', 'moment_8', 'moment_9',
                         'kstat_1', 'kstat_2', 'kstat_3', 'kstat_4',
                         'kstatvar_1', 'kstatvar_2']

    # Load signal data from the specified paths
    signals_data: List[np.ndarray] = [load_npy_file(path) for path in signal_paths]
    # Calculate statistical features for each loaded signal
    signal_summaries: List[Dict[str, Union[float, np.ndarray, int]]] = [get_features(sig, feature_names) for sig in tqdm(signals_data, desc="Calculating features")]
    # Convert the feature dictionaries into NumPy arrays
    signal_features: List[np.ndarray] = [dict_to_np(sig, feature_names) for sig in tqdm(signal_summaries, desc="Converting features to array")]
    
    # Discretize the features using KBinsDiscretizer
    # Note: This fits discretizers based on the current batch of signal_features
    if discretizers is None:
        # If no discretizers are provided, create new ones and fit them to the feature data
        discretizers: Dict[int, KBinsDiscretizer] = {}
        _, discretizers = discretize_features(np.array(signal_features), n_bins=10, strategy='uniform')
    # Apply the fitted discretizers to get the discrete representation for each signal summary
    signal_discretized_feature: List[Dict[str, Union[int, np.ndarray]]] = [get_discrete_info(sig, discretizers) for sig in tqdm(signal_summaries, desc="Discretizing features")]

    # Normalize features using standardization
    if scaler is None:
        # If no scaler is provided, create a new one and fit it to the feature data
        scaler = StandardScaler()
        _ = scaler.fit(signal_features) # Fit the scaler
    
    # Apply the scaler (either pre-fitted or newly fitted) to get scaled features
    signal_stats: List[Dict[str, Union[float, np.ndarray]]] = [get_scaled_info(sig, scaler) for sig in tqdm(signal_summaries, desc="Scaling features")]

    # Determine the unique options (classes) from the labels
    options: List[str] = list(set(signal_labels))

    # Generate the multiple-choice options string (e.g., "[A: opt1, B: opt2]").
    options_str: str = create_options(options)
    question_template_format: List[str] = [options_str]
    instruction_template_format: List[str] = []

    # Generate prompts using the scaled continuous features
    # These prompts might include few-shot examples if add_context is True
    context_prompts: List[str] = [
        generate_prompt(
            sig_info, question_template, question_template_format, instruction_template, instruction_template_format, feature_names, 
            processed=True, add_context=add_context, example_dict=example_dict, 
            decimal_precision=decimal_precision, options=options, 
            discretizers=None, scaler=scaler, discretized=False
        ) for sig_info in tqdm(signal_stats, desc="Generating continuous prompts")
    ]
    
    # Generate prompts using the discretized features (represented by letters)
    # These prompts might also include few-shot examples if add_context is True
    discret_context_prompts: List[str] = [
        generate_prompt(
            sig_info, question_template, question_template_format, instruction_template, instruction_template_format, feature_names, 
            processed=True, add_context=add_context, example_dict=example_dict, 
            decimal_precision=decimal_precision, options=options, 
            discretizers=discretizers, scaler=scaler, discretized=True
        ) for sig_info in tqdm(signal_discretized_feature, desc="Generating discrete prompts")
    ]

    # Compile all processed data and metadata into a dictionary
    data: Dict[str, Any] = {
        'signals': signals_data,                     # Raw signal data
        'stats': signal_stats,                       # Scaled continuous features (list of dicts)
        'discret_stats': signal_discretized_feature, # Discretized features (list of dicts)
        'labels': signal_labels,                     # Original labels
        'snrs': signal_snr,                          # Original SNRs
        'prompts': context_prompts,                  # Prompts with continuous features
        'discret_prompts': discret_context_prompts,  # Prompts with discrete features
        'feature_names': feature_names,              # List of feature names used
        'scaler': scaler,                            # Scaler object used
        'discretizers': discretizers,                # Discretizer objects used
        'num_samples': len(signal_labels),           # Total number of samples
        'num_features': signal_features[0].shape[0], # Number of features per sample
        '#classes': len(options),                    # Number of unique classes
        '#snr': len(set(signal_snr))                 # Number of unique SNRs
    }
    
    # Return the dictionary containing all processed information
    return data

# %%
def get_descrete_signal():
    pass

def get_family_example(family: Dict[str, Any], example_paths: Dict[str, str]) -> Dict[str,str]:
    """
    Generates a dictionary of example signal paths for each family in the family dictionary.
    This function takes a family dictionary where keys are family names and values are lists of example signal indices.
    It constructs a new dictionary where each key corresponds to a family name and the value is a list of file paths
    to the example signals. The file paths are obtained from the example_paths dictionary, which maps signal indices to file paths. 
    Args:
        family (Dict[str, Any]): A dictionary where keys are family names (str) and values are lists of example signal indices.
        example_paths (Dict[str, str]): A dictionary mapping signal indices (str) to file paths (str).
    Returns:
        Dict[str, str]: A dictionary where keys are family names (str) and values are lists of file paths (str) to the example signals.
    """
    family_example = {key: [] for key in family.keys()}
    for key in family.keys():
        ele = family[key]
        if isinstance(ele, dict):
            for k in ele.keys():
                for v in ele[k]:
                    family_example[key].extend(example_paths[v])
        elif isinstance(ele, list):
            for k in ele:
                family_example[key].extend(example_paths[k])
        else:
            # If the element is not a list or dict, treat it as a single example
            family_example[key].append(example_paths[ele])
    
    return family_example

def reduce_example_dict(
    example_dict: Dict[str, List[Any]], # Assuming values are lists of examples
    label: str,
    max_examples: int = 10
) -> Dict[str, List[Any]]:
    """
    Selects up to max_examples examples, ensuring diversity and label inclusion.

    This function selects a total of up to `max_examples` individual examples
    from the input dictionary `example_dict`. It prioritizes including at least
    one example from the category specified by `label`. It then tries to include
    one example from as many other distinct categories (keys) as possible.
    If more examples are needed to reach `max_examples`, they are chosen randomly
    from the remaining pool of available examples.

    Args:
        example_dict (Dict[str, List[Any]]): A dictionary where keys are category
                                            labels (str) and values are lists of
                                            corresponding examples (Any type).
        label (str): The specific category label (key) that must have at least
                     one example included in the output.
        max_examples (int, optional): The target maximum total number of individual
                                      examples in the returned dictionary. Defaults to 10.

    Returns:
        Dict[str, List[Any]]: A dictionary containing the selected examples, grouped
                              by their original keys. The values are lists of the
                              selected examples for that key. The total number of
                              examples across all lists will be at most `max_examples`.
                              Returns an empty dictionary if the input is empty or
                              contains only empty lists. Returns all examples grouped
                              by key if the total number available is less than or
                              equal to `max_examples`.

    Raises:
        ValueError: If `max_examples` is less than 1.
        ValueError: If the specified `label` is not found as a key in `example_dict`.
        ValueError: If the list associated with the `label` key is empty after filtering
                    keys with no examples.
    """
    if max_examples < 1:
        raise ValueError("max_examples must be at least 1.")

    if label not in example_dict:
        raise ValueError(f"Label '{label}' not found as a key in the example dictionary.")

    # Filter out keys with empty lists
    valid_example_dict = {k: v for k, v in example_dict.items() if v}

    if not valid_example_dict:
        return {} # Input dict was empty or contained only empty lists

    if label not in valid_example_dict:
         raise ValueError(f"Example list for label '{label}' is empty or the key was removed.")

    # Create a flat list of all available examples as (key, example_item) tuples
    all_examples: List[Tuple[str, Any]] = [
        (key, item) for key, items in valid_example_dict.items() for item in items
    ]

    # If total examples are less than or equal to requested, return all grouped by key
    if len(all_examples) <= max_examples:
        # print(f"Warning: Total available examples ({len(all_examples)}) is less than or equal to max_examples ({max_examples}). Returning all available examples grouped by key.")
        # Return the filtered dict containing all valid examples
        return valid_example_dict

    # --- Selection Process ---
    selected_examples_flat: List[Tuple[str, Any]] = []
    # Keep track of available examples using their indices in the all_examples list
    remaining_indices_pool = list(range(len(all_examples)))

    # 1. Select one example for the target label
    # Find indices in the pool that correspond to the label
    label_indices_in_pool = [
        idx for idx in remaining_indices_pool if all_examples[idx][0] == label
    ]
    # Choose one index randomly
    chosen_label_pool_idx = random.choice(label_indices_in_pool)
    # Add the corresponding example to our selection
    selected_examples_flat.append(all_examples[chosen_label_pool_idx])
    # Remove the chosen index from the pool
    remaining_indices_pool.remove(chosen_label_pool_idx)
    num_selected = 1
    keys_represented = {label} # Keep track of keys we've picked from

    # 2. Prioritize diversity: Select one example from other keys
    # Get keys other than the main label that have examples
    other_keys = [k for k in valid_example_dict.keys() if k != label]
    random.shuffle(other_keys) # Shuffle for random order

    # Create a map from key to list of pool indices for faster lookup during this phase
    key_to_pool_indices: Dict[str, List[int]] = {}
    for idx in remaining_indices_pool:
        key = all_examples[idx][0]
        if key not in key_to_pool_indices:
            key_to_pool_indices[key] = []
        key_to_pool_indices[key].append(idx)

    indices_to_remove_from_pool = [] # Store indices selected in this step
    for key in other_keys:
        if num_selected >= max_examples:
            break # Stop if we've reached the limit
        # Check if this key still has available examples in the pool
        if key in key_to_pool_indices and key_to_pool_indices[key]:
            # Choose a random pool index for this key
            chosen_pool_idx = random.choice(key_to_pool_indices[key])

            # Add the example to selection
            selected_examples_flat.append(all_examples[chosen_pool_idx])
            indices_to_remove_from_pool.append(chosen_pool_idx) # Mark for removal from pool
            keys_represented.add(key) # Mark key as represented
            num_selected += 1

            # Remove the chosen index from the lookup map to prevent re-picking it for this key
            key_to_pool_indices[key].remove(chosen_pool_idx)

    # Remove the selected indices from the main pool efficiently
    # Create a set for fast checking
    indices_removed_in_step2_set = set(indices_to_remove_from_pool)
    remaining_indices_pool = [idx for idx in remaining_indices_pool if idx not in indices_removed_in_step2_set]

    # 3. Fill remaining slots randomly if needed
    remaining_needed = max_examples - num_selected
    if remaining_needed > 0 and remaining_indices_pool:
        # Ensure we don't try to sample more than available
        num_to_sample = min(remaining_needed, len(remaining_indices_pool))
        # Randomly sample indices from the remaining pool
        randomly_chosen_pool_indices = random.sample(remaining_indices_pool, num_to_sample)

        # Add the corresponding examples to the selection
        for pool_idx in randomly_chosen_pool_indices:
             selected_examples_flat.append(all_examples[pool_idx])
        # num_selected += num_to_sample # Final count not strictly needed

    # 4. Reconstruct the dictionary from the flat list of selected examples
    new_example_dict: Dict[str, List[Any]] = {}
    for key, item in selected_examples_flat:
        if key not in new_example_dict:
            new_example_dict[key] = []
        new_example_dict[key].append(item)

    # Optional: Shuffle the lists within the dictionary if order matters
    # for key in new_example_dict:
    #     random.shuffle(new_example_dict[key])

    return new_example_dict

def get_family_label(signal_label: str, family: Dict[str, Any]) -> str:
    for key in family.keys():
        ele = family[key]
        if isinstance(ele, dict):
            for k in ele.keys():
                if signal_label in ele[k]:
                    return key
        elif isinstance(ele, list):
            if signal_label in ele:
                return key
    return ''
# %%

if __name__ == "__main__":
    # from templates import ( INPUT_TEXT, PROMPT_TEMPLATE, FEATURE_NAMES)
    # from radioml import (get_radioml_label, get_radioml_snr, radioml_example_maker)

    # train_signal_paths, example_paths, train_signal_labels, train_signal_snr = radioml_example_maker(pattern="../../data/RadioML/*/train/*/*.npy")

    # all_example_dict = {k:[p for p in v] for k,v in example_paths.items()}

    # r = (reduce_example_dict(all_example_dict, get_radioml_label(train_signal_paths[0]), max_examples=10))
    # print(len(r.keys()))
    # print(len(r[list(r.keys())[0]]))
    # options = list(set(train_signal_labels))
    # example_dict = {k:[load_npy_file(p) for p in v] for k,v in example_paths.items()}

    # train_signal_paths = train_signal_paths[:10]
 
    # feature_names = ['nobs', 'min', 'max', 'mean', 'variance', 'skewness', 'kurtosis', 
    #                      'moment_0', 'moment_1', 'moment_2', 'moment_3', 'moment_4', 
    #                      'moment_5', 'moment_6', 'moment_7', 'moment_8', 'moment_9',
    #                      'kstat_1', 'kstat_2', 'kstat_3', 'kstat_4',
    #                      'kstatvar_1', 'kstatvar_2']
    
    # train_data = get_processed_data(train_signal_paths, train_signal_labels, train_signal_snr, feature_names, INPUT_TEXT, PROMPT_TEMPLATE, example_dict, scaler=None, discretizers=None, decimal_precision=3, add_context=True)  
    
    # # Save the processed data to a file
    # save_processed_data(train_data, '../../data/RadioML/train_data.pkl')
    # print("Processed train data saved successfully.")

    # test_signal_paths = glob('../../data/RadioML/*/test/*/*.npy')
    # test_signal_labels = [get_radioml_label(sig) for sig in test_signal_paths]
    # test_signal_snr = [get_radioml_snr(sig) for sig in test_signal_paths]

    # test_signal_paths = test_signal_paths[:10]

    # test_data = get_processed_data(test_signal_paths, test_signal_labels, test_signal_snr, feature_names, INPUT_TEXT, PROMPT_TEMPLATE, example_dict, scaler=train_data['scaler'], discretizers=train_data['discretizers'], decimal_precision=3, add_context=True)

    # # Save the processed test data to a file
    # save_processed_data(test_data, '../../data/RadioML/test_data.pkl')
    # print("Processed test data saved successfully.")
    # from templates import ( CASCADE_PROMPT, MODULATION_FAMILIES, AMPLITUDE_FAMILY, ANANLOG_FAMILY, PHASE_FAMILY, FREQUENCY_FAMILY,)

    # all_example_dict = get_family_example(MODULATION_FAMILIES, example_paths)
    # example_dict = reduce_example_dict(all_example_dict, 'Amplitude-Based', max_examples=10)

    # print(example_dict)
    # print(len(example_dict['Amplitude-Based']))
    # print(len(MODULATION_FAMILIES.keys()))

    # print(get_family_example(AMPLITUDE_FAMILY, example_paths))
    # print(get_family_example(ANANLOG_FAMILY, example_paths))
    # print(get_family_example(PHASE_FAMILY, example_paths))
    # print(get_family_example(FREQUENCY_FAMILY, example_paths))



    

# %%
    # Our Dataset
    pass