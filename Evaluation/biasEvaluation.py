from Utils.paramsLoader import *
import torch
from DataPreprocessing.dataSplitter import splitDataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.utils import class_weight
# from DataPreprocessing.dataCollector import

def standaloneEval(
    params, test_data=None, extra_data_path=None, topk=2, use_ext_df=False
):
    device = torch.device("gpu")
    
    params_dash = {}
    params_dash["num_classes"] = 2
    params_dash["data_file"] = extra_data_path
    params_dash["class_names"] = dict_data_folder[str(params["num_classes"])][
        "class_label"
    ]
    temp_read = get_annotated_data(params_dash)
    with open("Data/post_id_divisions.json", "r") as fp:
        post_id_dict = json.load(fp)
    temp_read = temp_read[temp_read["post_id"].isin(post_id_dict["test"])]
    test_data = get_test_data(temp_read, params, message="text")
    test_extra = encodeData(test_data, vocab_own, params)
    test_dataloader = combine_features(test_extra, params, is_train=False)

def evaluate_bias():
    pass