from collections import defaultdict
import torch
import numpy as np
from knnbox.retriever.utils import retrieve_k_nearest
import torch.nn as nn


class Retriever:
    def __init__(self, datastore, k, save_knn_data=False, knn_xlingual_map=None):
        self.datastore = datastore
        self.k = k
        self.save_knn_data = save_knn_data
        self.results = None
        self.knn_data = None
        if self.save_knn_data:
            self.knn_data = defaultdict(list)
        
        self.xlingual_map = None
        if knn_xlingual_map not in [None, "None"]:
            assert torch.cuda.is_available()
            self.xlingual_map = torch.from_numpy(np.load(knn_xlingual_map)).cuda()
            print("Loaded cross-lingual map successfully.")


    def retrieve(self, query, return_list = ["vals", "distances"], k = None):
        r""" 
        retrieve the datastore, save and return results 
        if parameter k is provided, it will suppress self.k
        """
        k = k if k is not None else self.k
        # load the faiss index if haven't loaded
        if not hasattr(self.datastore, "faiss_index") or \
                    self.datastore.faiss_index is None or "keys" not in self.datastore.faiss_index:
            self.datastore.load_faiss_index("keys", move_to_gpu=True)

        query = query.detach()

        # apply cross-lingual mapping
        if self.xlingual_map is not None:
            query = query@self.xlingual_map

        faiss_results = retrieve_k_nearest(query, self.datastore.faiss_index["keys"], k)

        ret = {}
        if "distances" in return_list:
            ret["distances"] = faiss_results["distances"]
        if "indices" in return_list:
            ret["indices"] = faiss_results["indices"]
        if "k" in return_list:
            ret["k"] = k
        if "query" in return_list:
            ret["query"] = query

        # other information get from self.datastores.datas using indices, for example `keys` and `vals`
        indices = faiss_results["indices"].cpu().numpy()
        for data_name in return_list:
            if data_name not in ["distances", "indices", "k", "query"]:
                assert data_name in self.datastore.datas, \
                                    "You must load the {} of datastore first".format(data_name)
                ret[data_name] = torch.tensor(self.datastore[data_name].data[indices], device=query.device)
        
        self.results = ret # save the retrieved results

        # store all retrieved results
        if self.save_knn_data:
            for k in ret.keys():
                n_toks = ret[k].shape[0]*ret[k].shape[-1]
                self.knn_data[k].extend(list(ret[k].view(n_toks).cpu().numpy()))

        return ret
