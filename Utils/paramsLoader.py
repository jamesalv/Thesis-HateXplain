import json

dict_data_folder = {
    "2": {"data_file": "Data/dataset.json", "class_label": "Data/classes_two.npy"},
    "3": {"data_file": "Data/dataset.json", "class_label": "Data/classes.npy"},
}

model_dict_params = {
    "bert": "best_model_json/bestModel_bert_base_uncased_Attn_train_FALSE.json",
    "bert_supervised": "best_model_json/bestModel_bert_base_uncased_Attn_train_TRUE.json",
}

def load_params(path):
    with open(path, mode="r") as f:
        params = json.load(f)
    for key in params:
        if params[key] == "True":
            params[key] = True
        elif params[key] == "False":
            params[key] = False
        else:
            params[key] = params[key]

    return params

def return_params(path, att_lambda, num_classes=3, pretrained=False):
    with open(path, mode="r") as f:
        params = json.load(f)
    params = load_params(path)

    # Change att_lambda and num_classes manually
    params["att_lambda"] = att_lambda
    params["num_classes"] = num_classes

    # Load pretrained model path if any
    if pretrained:
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
            )
        else:
            output_dir = output_dir + "_" + str(params["num_classes"])
        params["pretrained_path"] = output_dir
    
    params["data_file"] = dict_data_folder[str(params["num_classes"])]["data_file"]
    params["class_names"] = dict_data_folder[str(params["num_classes"])]["class_label"]
    if params["num_classes"] == 2 and (params["auto_weights"] == False):
        params["weights"] = [1.0, 1.0]

    return params
