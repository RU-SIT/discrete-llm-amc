import re
from typing import Dict, List
import os

def acc(results: Dict[int, List[Dict]]):
    """Calculate accuracy"""
    correct_count = 0
    total_count = 0
    for key, value in results.items():
        true_label = value[0]['true_label']
        for v in value:
            if true_label in v['response_label']:
                correct_count += 1
            total_count += 1

    return (correct_count, total_count, correct_count / (total_count))

def find_classes_in_text(text, class_names):
    found_classes = []
    for class_name in class_names:
        if class_name in text:
            found_classes.append(class_name)
    return found_classes

def clean_acc(results: Dict[int, List[Dict]], class_names):
    """Calculate accuracy"""
    correct_count = 0
    total_count = 0
    for key, value in results.items():
        true_label = value[0]['true_label']
        for v in value:
            found_classes = find_classes_in_text(v['response_label'], class_names)
            if found_classes == []:
                continue
            elif true_label in found_classes:
                correct_count += 1
            total_count += 1

    return (correct_count, total_count, correct_count / (total_count))

def pass_acc(results: Dict[int, List[Dict]]):
    """Calculate pass accuracy"""
    pass_count = 0
    total_count = len(results)
    for key, value in results.items():
        true_label = value[0]['true_label']
        for v in value:
            if true_label in v['response_label']:
                pass_count += 1
                break

    return (pass_count, total_count, pass_count / (total_count))    

def majority_acc(results: Dict[int, List[Dict]]):
    pass_count = 0
    total_count = len(results)
    pass_criteria = len(results[0]) / 2
    for key, value in results.items():
        true_label = value[0]['true_label']
        key_pass_count = 0
        for v in value:
            if true_label in v['response_label']:
                key_pass_count += 1
        if key_pass_count > pass_criteria:
            pass_count += 1
            
    return (pass_count, total_count, pass_count / (total_count))

def create_folder(path):
    """Create folder if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def get_output_version(folder_path: str = "output"):
    """Get output version"""
    version = 0
    while os.path.exists(os.path.join(folder_path, f"version_{version}")):
        version += 1
    return version

def sort_results_by_prompt(results: List[Dict]) -> Dict[int, List[Dict]]:
    """Sort results by prompt and assign numeric keys"""
    sorted_results = {}
    prompts_to_id = {}  # Dictionary to map prompts to numeric IDs
    prompt_id = 0
    
    for result in results:
        prompt = result['filename']
        if prompt not in prompts_to_id:
            prompts_to_id[prompt] = prompt_id
            prompt_id += 1
        
        current_id = prompts_to_id[prompt]
        if current_id not in sorted_results:
            sorted_results[current_id] = []
        sorted_results[current_id].append(result)
    
    return sorted_results

def per_class_acc(results: Dict[int, List[Dict]], class_names: List[str]):
    """Calculate per-class accuracy after cleaning labels."""
    class_correct_count = {class_name: 0 for class_name in class_names}
    class_total_count = {class_name: 0 for class_name in class_names}

    for key, value in results.items():
        true_label = value[0]['true_label']
        if true_label not in class_names:
            continue

        for v in value:
            found_classes = find_classes_in_text(v['response_label'], class_names)
            if not found_classes:
                continue
            
            class_total_count[true_label] += 1
            if true_label in found_classes:
                class_correct_count[true_label] += 1

    class_accuracies = {}
    for class_name in class_names:
        if class_total_count[class_name] > 0:
            class_accuracies[class_name] = class_correct_count[class_name] / class_total_count[class_name]
        else:
            class_accuracies[class_name] = 0.0
            
    return class_accuracies


