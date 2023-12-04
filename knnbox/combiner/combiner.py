from collections import defaultdict
import torch
import torch.nn.functional as F

from knnbox.combiner.utils import calculate_knn_prob, calculate_combined_prob

class Combiner:
    r"""
    A simple Combiner used by vanilla knn-mt
    """

    def __init__(self, lambda_, temperature, probability_dim, save_top_k=False):
        self.lambda_ = lambda_
        self.temperature = temperature
        self.probability_dim = probability_dim
        
        # save data for analysis
        self.save_top_k = save_top_k
        if self.save_top_k:
            self.knn_data = defaultdict(list)

    def get_knn_prob(self, vals, distances, temperature=None, device="cuda:0", **kwargs):
        r"""
        calculate knn prob for vanilla knn-mt
        parameter temperature will suppress self.parameter
        """
        temperature = temperature if temperature is not None else self.temperature
        knn_probs = calculate_knn_prob(vals, distances, self.probability_dim, temperature, device, **kwargs)
        return knn_probs

    
    def get_combined_prob(self, knn_prob, neural_model_logit, lambda_ = None, log_probs = False):
        r""" 
        strategy of combine probability of vanilla knn-mt
        If parameter `lambda_` is given, it will suppress the self.lambda_ 
        """
        lambda_ = lambda_ if lambda_ is not None else self.lambda_

        combined_prob, extra = calculate_combined_prob(knn_prob, neural_model_logit, lambda_, log_probs)

        if self.save_top_k:
            self.knn_data["knn_top5_tok"].extend(list(knn_prob.topk(5)[1].squeeze().cpu().numpy()))
            self.knn_data["knn_top5_prob"].extend(list(knn_prob.topk(5)[0].squeeze().cpu().numpy()))
            self.knn_data["nn_top5_tok"].extend([list(top5) for top5 in extra["neural_probs"].topk(5)[1].squeeze().cpu().numpy()])
            self.knn_data["nn_top5_prob"].extend([list(top5) for top5 in extra["neural_probs"].topk(5)[0].squeeze().cpu().numpy()])
            self.knn_data["combi_top5_tok"].extend([list(top5) for top5 in combined_prob.topk(5)[1].squeeze().cpu().numpy()])
            self.knn_data["combi_top5_prob"].extend([list(top5) for top5 in combined_prob.topk(5)[0].squeeze().cpu().numpy()])

        return combined_prob, extra
        

        