import torch
import numpy as np
import json
from DataPreprocessing.dataSplitter import splitDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from DataPreprocessing.dataLoader import create_dataloaders
from Models.bert_weighted import Weighted_BERT
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from Utils.utils import *
import neptune
import time
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from transformers import BertTokenizer
from Utils.paramsLoader import *
# Parameters needed for training
params = []

def eval_phase(
    params, which_files="test", model=None, test_dataloader=None, device=None
):
    if params["is_model"] == True:
        print("model previously passed")
        model.eval()
    else:
        return 1

    print("Running eval on ", which_files, "...")
    t0 = time.time()

    true_labels = []
    pred_labels = []
    logits_all = []
    # Evaluate data for one epoch
    for step, batch in tqdm(enumerate(test_dataloader)):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention vals
        #   [2]: attention mask
        #   [3]: labels
        b_input_ids = batch[0].to(device)
        b_att_val = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)

        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()
        outputs = model(
            b_input_ids,
            attention_vals=b_att_val,
            attention_mask=b_input_mask,
            labels=None,
            device=device,
        )
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()
        # Calculate the accuracy for this batch of test sentences.
        # Accumulate the total accuracy.
        pred_labels += list(np.argmax(logits, axis=1).flatten())
        true_labels += list(label_ids.flatten())
        logits_all += list(logits)

    logits_all_final = []
    for logits in logits_all:
        logits_all_final.append(softmax(logits))

    testf1 = f1_score(true_labels, pred_labels, average="macro")
    testacc = accuracy_score(true_labels, pred_labels)
    if params["num_classes"] == 3:
        testrocauc = roc_auc_score(
            true_labels, logits_all_final, multi_class="ovo", average="macro"
        )
    else:
        # testrocauc=roc_auc_score(true_labels, logits_all_final,multi_class='ovo',average='macro')
        testrocauc = 0
    testprecision = precision_score(true_labels, pred_labels, average="macro")
    testrecall = recall_score(true_labels, pred_labels, average="macro")

    if params["logging"] != "neptune" or params["is_model"] == True:
        # Report the final accuracy for this validation run.
        print(" Accuracy: {0:.2f}".format(testacc))
        print(" Fscore: {0:.2f}".format(testf1))
        print(" Precision: {0:.2f}".format(testprecision))
        print(" Recall: {0:.2f}".format(testrecall))
        print(" Roc Auc: {0:.2f}".format(testrocauc))
        print(" Test took: {:}".format(format_time(time.time() - t0)))
        # print(ConfusionMatrix(true_labels,pred_labels))
    else:
        bert_model = params["path_files"]
        language = params["language"]
        name_one = bert_model + "_" + language
        neptune.create_experiment(
            name_one,
            params,
            send_hardware_metrics=False,
            run_monitoring_thread=False,
        )
        neptune.append_tag(bert_model)
        neptune.append_tag(language)
        neptune.append_tag("test")
        neptune.log_metric("test_f1score", testf1)
        neptune.log_metric("test_accuracy", testacc)
        neptune.log_metric("test_precision", testprecision)
        neptune.log_metric("test_recall", testrecall)
        neptune.log_metric("test_rocauc", testrocauc)
        neptune.stop()

    return testf1, testacc, testprecision, testrecall, testrocauc, logits_all_final


def train_model(params, device):
    # Split the dataset
    train, val, test = splitDataset(params)

    # Auto weights will do weighting based on the frequency of the class, this is done
    # to handle imbalance in the class
    if params["auto_weights"]:
        print(f"Classes: {np.unique(y_test)}")
        y_test = [row[-1] for row in test]
        encoder = LabelEncoder()
        encoder.classes_ = np.load(params["class_names"], allow_pickle=True)
        params["weights"] = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(y_test), y=y_test
        ).astype("float32")
    # Create dataloader
    dataloader = create_dataloaders(train, val, test, params)
    train_dataloader = dataloader["train"]
    val_dataloader = dataloader["val"]
    test_dataloader = dataloader["test"]

    # Modelling
    model = Weighted_BERT.from_pretrained(
        params["path_files"],
        num_labels=params["num_classes"],
        output_attentions=True,
        output_hidden_states=False,
        hidden_dropout_prob=params["dropout_bert"],
        params=params
    )

    model.cuda()
    optimizer = AdamW(
        model.parameters(), lr=params["learning_rate"], eps=params["epsilon"]
    )

    total_steps = len(train_dataloader) * params["epochs"]

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps / 10),
        num_training_steps=total_steps,
    )

    seed_all(seed=42)

    if params["logging"] == "neptune":
        neptune.create_experiment(
            params["path_files"],
            params=params,
            send_hardware_metrics=False,
            run_monitoring_thread=False,
        )

        neptune.append_tag(params["path_files"])
        if params["best_params"]:
            neptune.append_tag("AAAI final best")
        else:
            neptune.append_tag("AAAI final")

    best_val_fscore = 0
    best_test_fscore = 0

    best_val_roc_auc = 0
    best_test_roc_auc = 0

    best_val_precision = 0
    best_test_precision = 0

    best_val_recall = 0
    best_test_recall = 0
    loss_values = []

    for epoch_i in range(0, params["epochs"]):
        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, params["epochs"]))
        print("Training...")

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in tqdm(enumerate(train_dataloader)):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention vals
            #   [2]: attention mask
            #   [3]: labels
            b_input_ids = batch[0].to(device)
            b_att_val = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)

            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()
            outputs = model(
                b_input_ids,
                attention_vals=b_att_val,
                attention_mask=b_input_mask,
                labels=b_labels,
                device=device,
            )

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.

            loss = outputs[0]

            if params["logging"] == "neptune":
                neptune.log_metric("batch_loss", loss.item())
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        if params["logging"] == "neptune":
            neptune.log_metric("avg_train_loss", avg_train_loss)
        else:
            print("avg_train_loss", avg_train_loss)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        (
            train_fscore,
            train_accuracy,
            train_precision,
            train_recall,
            train_roc_auc,
            _,
        ) = eval_phase(params, "train", model, train_dataloader, device)
        val_fscore, val_accuracy, val_precision, val_recall, val_roc_auc, _ = (
            eval_phase(params, "val", model, val_dataloader, device)
        )
        (
            test_fscore,
            test_accuracy,
            test_precision,
            test_recall,
            test_roc_auc,
            logits_all_final,
        ) = eval_phase(params, "test", model, test_dataloader, device)

        # Report the final accuracy for this validation run.
        if params["logging"] == "neptune":
            neptune.log_metric("test_fscore", test_fscore)
            neptune.log_metric("test_accuracy", test_accuracy)
            neptune.log_metric("test_precision", test_precision)
            neptune.log_metric("test_recall", test_recall)
            neptune.log_metric("test_rocauc", test_roc_auc)

            neptune.log_metric("val_fscore", val_fscore)
            neptune.log_metric("val_accuracy", val_accuracy)
            neptune.log_metric("val_precision", val_precision)
            neptune.log_metric("val_recall", val_recall)
            neptune.log_metric("val_rocauc", val_roc_auc)

            neptune.log_metric("train_fscore", train_fscore)
            neptune.log_metric("train_accuracy", train_accuracy)
            neptune.log_metric("train_precision", train_precision)
            neptune.log_metric("train_recall", train_recall)
            neptune.log_metric("train_rocauc", train_roc_auc)

        if val_fscore > best_val_fscore:
            print(val_fscore, best_val_fscore)
            best_val_fscore = val_fscore
            best_test_fscore = test_fscore
            best_val_roc_auc = val_roc_auc
            best_test_roc_auc = test_roc_auc

            best_val_precision = val_precision
            best_test_precision = test_precision
            best_val_recall = val_recall
            best_test_recall = test_recall

            print("Loading BERT tokenizer...")
            tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased", do_lower_case=False
            )
            save_bert_model(model, tokenizer, params)

    if params["logging"] == "neptune":
        neptune.log_metric("best_val_fscore", best_val_fscore)
        neptune.log_metric("best_test_fscore", best_test_fscore)
        neptune.log_metric("best_val_rocauc", best_val_roc_auc)
        neptune.log_metric("best_test_rocauc", best_test_roc_auc)
        neptune.log_metric("best_val_precision", best_val_precision)
        neptune.log_metric("best_test_precision", best_test_precision)
        neptune.log_metric("best_val_recall", best_val_recall)
        neptune.log_metric("best_test_recall", best_test_recall)

        neptune.stop()
    else:
        print("best_val_fscore", best_val_fscore)
        print("best_test_fscore", best_test_fscore)
        print("best_val_rocauc", best_val_roc_auc)
        print("best_test_rocauc", best_test_roc_auc)
        print("best_val_precision", best_val_precision)
        print("best_test_precision", best_test_precision)
        print("best_val_recall", best_val_recall)
        print("best_test_recall", best_test_recall)

    del model
    torch.cuda.empty_cache()
    return 1


def train(path, custom_params):
    # Loading params
    params = load_params(path)
    params['path'] = 'Data/dataset.json'
    
    for key, value in custom_params.items():
        params[key] = value
    print(f"Training parameters: {params}")

    # Set device
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.device.__name__}")
        device = torch.device("cuda")
    else:
        print(f"GPU not available, using CPU instead !")
        device = torch.device("cpu")

    train_model(params, device)
