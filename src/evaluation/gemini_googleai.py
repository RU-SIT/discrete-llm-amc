from cmd import PROMPT
import os
import json
import pickle
import traceback
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List
from dotenv import load_dotenv
import google.generativeai as genai


from utils import pass_acc, acc, majority_acc, clean_acc, per_class_acc
load_dotenv()

def get_gemini_response(prompt: str, model: str = "gemini-1.5-pro-latest", temperature: float = 0.7) -> str:
    """
    Get response from Gemini 2.5 Pro API.
    """
    if type(model) is str:
        model_obj = genai.GenerativeModel(model)
    else:
        model_obj = model
    try:
        response = model_obj.generate_content(prompt, generation_config={"temperature": temperature})
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return ""
    
def extract_think_text(response: str) -> str:
    """Extract text between <think> tags"""
    start_tag = "<think>"
    end_tag = "</think>"
    start_idx = response.find(start_tag)
    end_idx = response.find(end_tag)
    
    if start_idx != -1 and end_idx != -1:
        return end_idx+len(end_tag), response[start_idx + len(start_tag):end_idx].strip()
    return 0, ""

def sample_per_label(data, per_label=20, seed=42, shuffle=True):
    """
    Sample up to `per_label` indices per label from `data['labels']`.
    Returns (sampled_data, selected_indices, rng).
    """
    rng = np.random.default_rng(seed)

    # Group indices by label
    indices_by_label = defaultdict(list)
    for i, label in enumerate(data['labels']):
        indices_by_label[label].append(i)

    # Sample from each label group
    selected_indices = []
    for _, indices in indices_by_label.items():
        num_to_sample = min(per_label, len(indices))
        sampled = rng.choice(indices, size=num_to_sample, replace=False)
        selected_indices.extend(int(i) for i in sampled)

    # Optionally shuffle
    if shuffle:
        rng.shuffle(selected_indices)

    # Build sampled_data, preserving non-list/non-array entries
    sampled_data = {}
    for key, value in data.items():
        if isinstance(value, (list, np.ndarray)):
            sampled_data[key] = [value[i] for i in selected_indices if i < len(value)]
        else:
            sampled_data[key] = value

    # Info
    print(f"Total samples selected: {len(sampled_data.get('labels', []))}")
    print(f"Sampled data keys: {list(sampled_data.keys())}")
    unique_labels, counts = np.unique(sampled_data.get('labels', []), return_counts=True)
    print("Label distribution in sampled data:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")

    return sampled_data, selected_indices, rng

def main(prompt_type='discret_prompts', model_name="gemini-2.5-flash", noise_mode='noisySignal', n_bins=10, top_k=5, num_tries=3):
    try:
        # # Load training data
        with open(f'../../data/own/unlabeled_10k/test_{noise_mode}_{n_bins}_{top_k}_data.pkl', 'rb') as f:
            whole_data = pickle.load(f)

        data, selected_indices, rng = sample_per_label(whole_data, per_label=20, seed=42, shuffle=True)
        # data = whole_data

        # with open("./first round exp/discret_noisy_gemini-2.5-flash_gemini_responses.json", "r") as f:
        #     whole_data = json.load(f)

        # data = {'discret_prompts': [r['prompt'] for r in whole_data if r['try']==0], 'signal_paths': [r['filename'] for r in whole_data if r['try']==0]}

        # Initialize OpenAI client
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)

        # Create a list of dictionaries, each containing the prompt and its corresponding filename
        all_prompts_data = []
        for i, prompt in enumerate(data[prompt_type][:332]):
            filename = os.path.basename(data['signal_paths'][i])
            all_prompts_data.append({'prompt': prompt, 'filename': filename})
        print(len(all_prompts_data))
        
        prompts_to_process = []
        if os.path.exists(f"{prompt_type}_{model_name}_{noise_mode}_{n_bins}_{top_k}_gemini_responses.json"):
            with open(f"{prompt_type}_{model_name}_{noise_mode}_{n_bins}_{top_k}_gemini_responses.json", 'r') as f:
                try:
                    results = json.load(f)
                    if not isinstance(results, list): results = []
                except (json.JSONDecodeError, TypeError):
                    results = []
            print(f"Loaded {len(results)} existing results.")
            
            if results:
                # Count valid tries for each prompt
                prompts_done = {}
                valid_results = [r for r in results if r.get('raw_response')]
                for r in valid_results:
                    prompts_done[r['filename']] = prompts_done.get(r['filename'], 0) + 1

                # Identify fully completed prompts
                completed_prompts = {p for p, count in prompts_done.items() if count >= num_tries}
                print(f"Found {len(completed_prompts)} fully completed prompts.")

                # Filter results to only keep results from fully completed prompts
                results = [r for r in results if r['filename'] in completed_prompts]
                print(f"Keeping {len(results)} results from completed prompts.")

                # Create the list of prompts to process (those not completed)
                for prompt_data in all_prompts_data:
                    if prompt_data['filename'] not in completed_prompts:
                        num_done = prompts_done.get(prompt_data['filename'], 0)
                        prompts_to_process.append((prompt_data, num_done))
                
                print(f"Identified {len(prompts_to_process)} prompts to process.")
            else:
                # No valid results found, process everything
                results = []
                prompts_to_process = [(p_data, 0) for p_data in all_prompts_data]
        else:
            results = []
            prompts_to_process = [(p_data, 0) for p_data in all_prompts_data]

        # Process each signal and store results
        model_obj = genai.GenerativeModel(model_name)
        for prompt_data, num_done in tqdm(prompts_to_process, total=len(prompts_to_process)):
            try:
                prompt = prompt_data['prompt']
                filename = prompt_data['filename']
                # Run only the remaining number of tries needed
                for i in range(num_done, num_tries):
                    raw_response = get_gemini_response(prompt, model=model_obj, temperature=0.2)

                    if raw_response == "":
                        raise Exception("reached limits of Gemini API, stopping")
                    
                    # Extract true label from filename
                    true_label = filename.split('_')[0]

                    end_idx, reasoning = extract_think_text(raw_response)

                    # Store result
                    result = {
                        'filename': filename,
                        'try': i,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "true_label": true_label,
                        "reasoning": reasoning,
                        "response_label": raw_response[end_idx:]
                    }
                    results.append(result)
                
            except Exception as e:
                if '"reached limits of Gemini API, stopping"' in str(e):
                    print("Reached limits of Gemini API, stopping further requests.")
                    break

                print(f"Error processing {filename}: {str(e)}")
                print(traceback.format_exc())

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving partial results...")

    except Exception as e:
        print(traceback.format_exc())

    finally:
        # Save results to JSON file using a temporary file for robustness
        temp_filepath = f"{prompt_type}_{model_name}_{noise_mode}_{n_bins}_{top_k}_gemini_responses.json.tmp"
        final_filepath = f"{prompt_type}_{model_name}_{noise_mode}_{n_bins}_{top_k}_gemini_responses.json"
        try:
            with open(temp_filepath, 'w') as f:
                json.dump(results, f, indent=4)
            # If write is successful, atomically rename the temp file to the final one
            os.rename(temp_filepath, final_filepath)
            print(f"Results saved to {final_filepath}")
        except Exception as e:
            print(f"CRITICAL: Failed to save results to {final_filepath}. Error: {e}")


def read_results(prompt_type, model_name, noise_mode, n_bins, top_k):
    """Read results from JSON file"""
    with open(f"{prompt_type}_{model_name}_{noise_mode}_{n_bins}_{top_k}_gemini_responses.json", 'r') as f:
        results = json.load(f)

    if model_name == "gemini-2.5-flash":
        # For Gemini 2.5 Flash, we need to clean up the response labels
        results, problematic_data = clean_up_response_label(results)
        if problematic_data:
            print(f"Found {len(problematic_data)} problematic data samples with long response labels.")
            print("These samples will be saved in the results with their raw responses.")
    return results

def get_unique_prompts(results: List[Dict]) -> List[str]:
    """Get unique prompts from results"""
    return list(set(result['filename'] for result in results))

def sort_results_by_prompt(results: List[Dict]) -> Dict[int, List[Dict]]:
    """Sort results by prompt and assign numeric keys"""
    sorted_results = {}
    prompts_to_id = {}  # Dictionary to map prompts to numeric IDs
    prompt_id = 0
    
    # Ensure prompts are sorted according to their first appearance in the results
    # This makes the sorting deterministic
    all_prompts_in_results = list(dict.fromkeys(r['filename'] for r in results))

    for prompt in all_prompts_in_results:
        if prompt not in prompts_to_id:
            prompts_to_id[prompt] = prompt_id
            prompt_id += 1

    for result in results:
        prompt = result['filename']
        current_id = prompts_to_id[prompt]
        if current_id not in sorted_results:
            sorted_results[current_id] = []
        sorted_results[current_id].append(result)
    
    return sorted_results

def clean_up_response_label(results: List[Dict]) -> (List[Dict], List[Dict]):
    """
    Clean up the response labels in the results by extracting valid class names.
    Returns the cleaned results and a list of problematic data samples.
    """
    problematic_data = []
    for sample in results:
        if len(sample['response_label']) > 10:
            problematic_data.append(sample)
            sample['response_label'] = sample['raw_response'][-6:]
    return results, problematic_data

if __name__ == '__main__':
    PROMPT_TYPE = 'discret_prompts'  # Options: 'discret_prompts', 'prompts', 'old_prompts', 'old_discret_prompts'
    MODEL_NAME = "gemini-2.5-pro" # Options: "gemini-2.5-pro", "gemini-2.5-flash"
    NOISE_MODE = 'noisySignal' # Options: 'noisySignal', 'noiselessSignal'
    N_BINS = 5
    TOP_K = 5
    NUM_TRIES = 1
    main(PROMPT_TYPE, MODEL_NAME, NOISE_MODE, N_BINS, TOP_K, NUM_TRIES)
    results = read_results(PROMPT_TYPE, MODEL_NAME, NOISE_MODE, N_BINS, TOP_K)
    unique_prompts = get_unique_prompts(results)
    print(f"Number of unique prompts: {len(unique_prompts)}")
    sorted_results = sort_results_by_prompt(results)
    print(f"Number of unique prompts: {len(sorted_results)}")
    print(f"acc", acc(sorted_results))
    class_names = ['4ASK', '4PAM', '8ASK', '16PAM', 'CPFSK', 'DQPSK', 'GFSK', 'GMSK', 'OQPSK', 'OOK']
    print(f"clean-acc", clean_acc(sorted_results, class_names=class_names))
    print(f"{len(sorted_results[0])}-pass", pass_acc(sorted_results))
    print(f"{len(sorted_results[0])}-majority", majority_acc(sorted_results))
    print(f"per-class-acc", per_class_acc(sorted_results, class_names=class_names))