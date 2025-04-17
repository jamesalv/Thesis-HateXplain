import numpy as np
from math import exp
import torch
import torch.nn as nn
import random
import datetime
import os

# General utility function
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return list(e_x / e_x.sum(axis=0))


def neg_softmax(x):
    """Compute softmax values for each sets of scores in x. Here we convert the exponentials to 1/exponentials"""
    e_x = np.exp(-(x - np.max(x)))
    return list(e_x / e_x.sum(axis=0))


def sigmoid(z):
    """Compute sigmoid values"""
    g = 1 / (1 + exp(-z))
    return list(g)

def cross_entropy(input1, target, size_average=True):
    """Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax(dim=0)
    return torch.sum(-target * logsoftmax(input1))

def masked_cross_entropy(input1, target, mask):
    cr_ent = 0
    for h in range(0, mask.shape[0]):
        cr_ent += cross_entropy(input1[h][mask[h]], target[h][mask[h]])

    return cr_ent / mask.shape[0]

def seed_all(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def save_bert_model(model, tokenizer, params):
    output_dir = "Saved/" + params["path_files"] + "_"
    if params["train_att"]:
        if params["att_lambda"] >= 1:
            params["att_lambda"] = int(params["att_lambda"])

        output_dir = (
            output_dir
            + str(params["supervised_layer_pos"])
            + "_"
            + str(params["num_supervised_heads"])
            + "_"
            + str(params["num_classes"])
            + "_"
            + str(params["att_lambda"])
            + "/"
        )

    else:
        output_dir = output_dir + "_" + str(params["num_classes"]) + "/"
    print(output_dir)
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
def fix_the_random(seed_val=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)