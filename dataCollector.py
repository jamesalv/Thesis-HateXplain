import pandas as pd
import json
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Union, Any
from transformers import BertTokenizer
from textPreprocess import ek_extra_preprocess
from utils import softmax, neg_softmax, sigmoid


def process_rationale_mask(text_tokens: List[str], mask: List[int]) -> Tuple[List[str], List[int]]:
    """
    Process a rationale mask to identify contiguous segments of highlighted text.
    
    Args:
        text_tokens: Original text tokens
        mask: Binary mask where 1 indicates a highlighted token
        
    Returns:
        A tuple of (text segments, corresponding masks)
    """
    # Handle case where mask is [-1, -1, ...] (no rationale provided)
    if mask[0] == -1:
        mask = [0] * len(mask)
    
    # Find breakpoints (transitions between highlighted and non-highlighted)
    breakpoints = []
    mask_values = []
    
    # Always start with position 0
    breakpoints.append(0)
    mask_values.append(mask[0])
    
    # Find transitions in the mask
    for i in range(1, len(mask)):
        if mask[i] != mask[i-1]:
            breakpoints.append(i)
            mask_values.append(mask[i])
    
    # Always end with the length of the text
    if breakpoints[-1] != len(mask):
        breakpoints.append(len(mask))
    
    # Create segments based on breakpoints
    segments = []
    for i in range(len(breakpoints) - 1):
        start = breakpoints[i]
        end = breakpoints[i+1]
        segments.append((text_tokens[start:end], mask_values[i]))
    
    return segments


def tokenize_and_mask(segments: List[Tuple[List[str], int]], 
                      params: Dict[str, Any], 
                      tokenizer) -> Tuple[List[int], List[int]]:
    """
    Tokenize text segments and create corresponding mask.
    
    Args:
        segments: List of (text segment, mask value) pairs
        params: Configuration parameters
        tokenizer: BERT tokenizer
        
    Returns:
        Tuple of (token_ids, token_masks)
    """
    # Start with special tokens if using BERT
    if params["bert_tokens"]:
        token_ids = [101]  # CLS token
        token_mask = [0]   # No attention on CLS
    else:
        token_ids = []
        token_mask = []
    
    # Process each segment
    for segment_text, mask_value in segments:
        # Tokenize the segment
        segment_str = " ".join(segment_text)
        tokens = ek_extra_preprocess(segment_str, params, tokenizer)
        # Apply same mask value to all tokens in this segment
        masks = [mask_value] * len(tokens)
        
        token_ids.extend(tokens)
        token_mask.extend(masks)
    
    # Add end token if using BERT and truncate if needed
    if params["bert_tokens"]:
        max_len = int(params["max_length"]) - 2
        token_ids = token_ids[:max_len]
        token_mask = token_mask[:max_len]
        token_ids.append(102)  # SEP token
        token_mask.append(0)   # No attention on SEP
    
    return token_ids, token_mask


def return_mask(row: Dict[str, Any], tokenizer, params: Dict[str, Any]) -> Tuple[List[int], List[List[int]]]:
    """
    Process the text and rationales to create token IDs and attention masks.
    
    Args:
        row: Data row containing text and rationales
        tokenizer: BERT tokenizer
        params: Configuration parameters
        
    Returns:
        Tuple of (token_ids, attention_masks)
    """
    text_tokens = row['text']
    masks = row["rationales"]
    
    word_mask_all = []
    word_tokens = None
    
    for mask in masks:
        # Process each mask to get segments
        segments = process_rationale_mask(text_tokens, mask)
        
        # Tokenize segments and create mask
        token_ids, token_mask = tokenize_and_mask(segments, params, tokenizer)
        
        # Store the first token_ids we generate
        if word_tokens is None:
            word_tokens = token_ids
            
        word_mask_all.append(token_mask)
    
    return word_tokens, word_mask_all[:len(masks)]


def aggregate_attention(attention_masks: List[List[int]], row: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    """
    Aggregate attention from different annotators into a single attention vector.
    
    Args:
        attention_masks: List of attention masks from annotators
        row: Data row with label information
        params: Configuration parameters
        
    Returns:
        Aggregated attention vector
    """
    # Convert to numpy array for easier operations
    attention_array = np.array(attention_masks)
    
    # If the label is normal/non-toxic, use uniform attention
    if row['final_label'] in ['normal', 'non-toxic']:
        mask_length = len(attention_masks[0])
        return np.ones(mask_length) / mask_length
    
    # Apply variance scaling
    scaled_attention = int(params['variance']) * attention_array
    
    # Average across annotators
    mean_attention = np.mean(scaled_attention, axis=0)
    
    # Apply normalization based on parameter setting
    if params['type_attention'] == 'sigmoid':
        return sigmoid(mean_attention)
    elif params['type_attention'] == 'softmax':
        return softmax(mean_attention)
    elif params['type_attention'] == 'neg_softmax':
        return neg_softmax(mean_attention)
    elif params['type_attention'] in ['raw', 'individual']:
        return mean_attention
    
    # Default fallback
    return mean_attention


def load_raw_data(file_path: str) -> Dict[str, Any]:
    """
    Load the raw JSON data from file.
    
    Args:
        file_path: Path to the JSON dataset file
        
    Returns:
        Loaded JSON data
    """
    with open(file_path, "r") as file:
        return json.load(file)


def preprocess_entry(key: str, value: Dict[str, Any], params: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """
    Preprocess a single data entry.
    
    Args:
        key: Post ID
        value: Post data
        params: Configuration parameters
        tokenizer: BERT tokenizer
        
    Returns:
        Preprocessed data entry or None if entry should be skipped
    """
    data_dict = {"post_id": key, "text": value["post_tokens"]}
    
    # Extract labels and target groups
    labels = [annot["label"] for annot in value["annotators"]]
    target_groups = []
    for annot in value["annotators"]:
        target_groups.extend(annot["target"])
    
    # Skip entries where all annotators disagree
    if len(set(labels)) == 3:
        return None
    
    # Determine final label
    data_dict["final_label"] = Counter(labels).most_common()[0][0]
    
    # Convert to binary classification if needed
    if params['num_classes'] == 2:
        if data_dict['final_label'] in ('hatespeech', 'offensive'):
            data_dict['final_label'] = 'toxic'
    
    # Process target groups (remove duplicates)
    data_dict["target_groups"] = list(set(target_groups))
    
    # Ensure there are 3 rationales
    rationales = value.get('rationales', [])
    while len(rationales) < 3:
        rationales.append([0] * len(value["post_tokens"]))
    data_dict['rationales'] = rationales
    
    # Get tokens and attention masks
    tokens, attention_masks = return_mask(data_dict, tokenizer, params)
    
    # Aggregate attention
    attention_vector = aggregate_attention(attention_masks, data_dict, params)
    
    data_dict['text_vector'] = tokens
    data_dict['attention'] = attention_vector
    
    return data_dict


def load_and_preprocess_data(params: Dict[str, Any], tokenizer) -> pd.DataFrame:
    """
    Load and preprocess the dataset.
    
    Args:
        params: Configuration parameters
        tokenizer: BERT tokenizer
        
    Returns:
        DataFrame containing preprocessed data
    """
    from tqdm import tqdm
    
    # Load raw data
    data = load_raw_data(params['path'])
    
    all_data = []
    count_confused = 0
    
    # Process each entry with progress bar
    for key, value in tqdm(data.items(), desc="Processing entries", unit="entry"):
        processed_entry = preprocess_entry(key, value, params, tokenizer)
        
        if processed_entry is None:
            count_confused += 1
            continue
            
        all_data.append(processed_entry)
    
    # Print statistics
    print(f"Initial data: {len(data)}")
    print(f"Uncertain data: {count_confused}")
    print(f"Total final data count: {len(all_data)}")
    
    return pd.DataFrame(all_data)


def collect_data(params: Dict[str, Any]) -> pd.DataFrame:
    """
    Main function to collect and preprocess data.
    
    Args:
        params: Configuration parameters
        
    Returns:
        DataFrame with preprocessed data
    """
    import os
    import pickle
    from tqdm import tqdm
    
    # Create a cache filename based on the parameters that affect preprocessing
    cache_dir = params.get('cache_dir', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a unique cache filename based on key parameters
    cache_params = {
        'path': params['path'],
        'num_classes': params['num_classes'],
        'bert_tokens': params['bert_tokens'],
        'max_length': params['max_length'],
        'type_attention': params['type_attention'],
        'variance': params['variance']
    }
    cache_hash = hash(frozenset(cache_params.items()))
    cache_file = os.path.join(cache_dir, f"preprocessed_data_{cache_hash}.pkl")
    
    # Check if cached data exists
    if os.path.exists(cache_file) and not params.get('force_preprocess', False):
        print(f"Loading preprocessed data from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print("Preprocessing data from scratch...")
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=False
    )
    
    data = load_and_preprocess_data(params, tokenizer)
    
    # Save processed data to cache
    print(f"Saving preprocessed data to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    return data