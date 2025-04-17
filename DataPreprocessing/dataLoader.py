from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.utils import pad_sequences


def create_dataloaders(train_df, val_df, test_df, params):
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        train_df: Training dataframe with 'text_vector', 'attention', and 'final_label'
        val_df: Validation dataframe
        test_df: Test dataframe
        params: Configuration parameters
        
    Returns:
        Dictionary containing train, validation, and test dataloaders
    """
    # Load the label encoder
    encoder = LabelEncoder()
    encoder.classes_ = np.load(params["class_names"], allow_pickle=True)
    
    # print(train_df.head())
    
    # Create dataloaders
    train_loader = create_single_dataloader(train_df, encoder, params, is_train=True)
    val_loader = create_single_dataloader(val_df, encoder, params, is_train=False)
    test_loader = create_single_dataloader(test_df, encoder, params, is_train=False)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

def create_single_dataloader(df, encoder, params, is_train=False):
    """
    Create a single dataloader from a dataframe.
    
    Args:
        df: Dataframe with 'text_vector', 'attention', and 'final_label'
        encoder: Label encoder
        params: Configuration parameters
        is_train: Whether this is for training (affects sampling strategy)
        
    Returns:
        PyTorch DataLoader
    """
    # Extract features
    input_ids = df['text_vector'].tolist()
    att_vals = df['attention'].tolist()
    labels = encoder.transform(df['final_label'].tolist())
    
    # Pad sequences
    input_ids = pad_sequences(
        input_ids,
        maxlen=int(params["max_length"]),
        dtype="long",
        value=0,
        truncating="post",
        padding="post",
    )
    
    # Pad attention values
    att_vals = pad_sequences(
        att_vals,
        maxlen=int(params["max_length"]),
        dtype="float",
        value=0.0,
        truncating="post",
        padding="post",
    )
    
    # Create padding masks (1 for real tokens, 0 for padding)
    padding_masks = [[int(token_id > 0) for token_id in seq] for seq in input_ids]
    
    # Convert to tensors
    inputs = torch.tensor(input_ids)
    attention = torch.tensor(np.array(att_vals), dtype=torch.float)
    masks = torch.tensor(np.array(padding_masks), dtype=torch.uint8)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Create dataset and dataloader
    data = TensorDataset(inputs, attention, masks, labels)
    
    if is_train:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
        
    return DataLoader(data, sampler=sampler, batch_size=params["batch_size"])