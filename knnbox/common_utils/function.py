from collections import defaultdict
import json
import os
import faiss
import numpy as np
import time
import pickle
import torch


_global_vars = {}
def global_vars():
    return _global_vars


def read_config(path):
    r"""
    read the config file under the `path` folder

    Args:
        path:
            folder where the config file is stored
    
    Returns:
        dict
    """
    config_file = os.path.join(path, "config.json")
    with open(config_file, encoding="utf-8", mode="r") as f:
        return json.load(f)


def write_config(path, config):
    r"""
    write the config file to the `path` folder

    Args:
        path:
            folder where the config file is stored
    
    Returns:
        dict
    """
    with open(os.path.join(path, "config.json"), encoding="utf-8", mode="w") as f:
        json.dump(config, f, indent = 6)


def filter_pad_tokens(tokens, pad_idx=1):
    r"""
    given a int tensor, 
    return all no pad element and the mask,
    1 represent no-pad, 0 represent pad
    """
    mask = tokens.ne(pad_idx)
    tokens = tokens.masked_select(mask)
    return tokens, mask



def select_keys_with_pad_mask(keys, mask):
    r"""
    use the mask to chose keys 

    Args:
        keys: (batch_sz, seq, dim)
        mask: (batch_sz, seq)
    
    Return: (*, dim)
    """
    mask_shape = mask.size()
    mask = mask.unsqueeze(-1).repeat(*([1]*len(mask_shape)+[keys.size(-1)]))
    return keys.masked_select(mask).view(-1, keys.size(-1))


def disable_model_grad(model):
    r""" disable whole model's gradient """
    for name, param in model.named_parameters():
        param.requires_grad = False


def enable_module_grad(model, module_name):
    r""" enable a module's gridient caclulation by module name"""
    for name, param in model.named_parameters():
        if module_name in name:
            param.requires_grad = True

def label_smoothed_nll_loss(lprobs, target, epsilon=2e-3, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss 

def save_knn_vals_to_disk(src_dict, tgt_dict, retriever, combiner, file_name):
    r"""Serialize and save the KNN results to disk.

    Args:
        src_dict (dict): A dictionary mapping integer IDs to source language tokens.
        tgt_dict (dict): A dictionary mapping integer IDs to target language tokens.
        retriever (Retriever): A kNN-MT Retriever object
        combiner (Combiner): A kNN-MT Combiner object
        file_name (str): The name of the file to write the serialized results to.
    """
    # get all relevant results stored in Retriever
    assert retriever.knn_data
    assert all([len(retriever.knn_data[k]) == len(retriever.knn_data["vals"]) for k in retriever.knn_data.keys()])
    retriever.knn_data["vals"] = [tgt_dict[x] for x in retriever.knn_data["vals"]]
    retriever.knn_data["tgt_lang_tags"] = [tgt_dict[x] for x in retriever.knn_data["tgt_lang_tags"]]
    retriever.knn_data["src_lang_tags"] = [src_dict[x] for x in retriever.knn_data["src_lang_tags"]]

    # get all relevant results stored in Combiner
    assert combiner.knn_data
    assert all([len(combiner.knn_data[k]) == len(combiner.knn_data["knn_top5_prob"]) for k in combiner.knn_data.keys()])
    # changes indices to tokens
    combiner.knn_data["knn_top5_tok"] = [[tgt_dict[t] for t in knn_top5] for knn_top5 in combiner.knn_data["knn_top5_tok"]]
    combiner.knn_data["nn_top5_tok"] = [[tgt_dict[t] for t in nn_top5] for nn_top5 in combiner.knn_data["nn_top5_tok"]]
    combiner.knn_data["combi_top5_tok"] = [[tgt_dict[t] for t in combi_top5] for combi_top5 in combiner.knn_data["combi_top5_tok"]]

    # combine and save results
    knn_data = defaultdict(list, {**retriever.knn_data, **combiner.knn_data})
    with open(file_name, "wb") as f:
        pickle.dump(knn_data, f)
