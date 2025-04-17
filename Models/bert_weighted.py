import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from Utils.utils import masked_cross_entropy


class Weighted_BERT(BertPreTrainedModel):
    def __init__(self, config, params):
        print(f"Config Params: {config}")
        print(f"Model Params: {params}")
        super().__init__(config)
        self.num_labels = config.num_labels
        # Parse string weights if needed
        if isinstance(params["weights"], str):
            import ast
            self.weights = ast.literal_eval(params["weights"])
        else:
            self.weights = params["weights"]
        self.train_att = params["train_att"]
        self.att_lambda = params["att_lambda"]
        self.num_sv_heads = params["num_supervised_heads"]
        self.sv_layer_pos = params["supervised_layer_pos"]
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        attention_vals=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        device=None,
    ):
        # Add debug print statements
        print(f"DEBUG - self.weights type: {type(self.weights)}")
        print(f"DEBUG - self.weights value: {self.weights}")
        print(f"DEBUG - self.num_labels: {self.num_labels}")

        # Try to convert weights to tensor and print
        try:
            weight_tensor = torch.tensor(self.weights)
            print(f"DEBUG - weight tensor shape: {weight_tensor.shape}")
        except Exception as e:
            print(f"DEBUG - Error converting weights to tensor: {e}")

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_funct = nn.CrossEntropyLoss(
                weight=torch.tensor(self.weights).to(device)
            )
            loss_logits = loss_funct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss_logits
            if self.train_att:
                loss_att = 0
                for i in range(self.num_sv_heads):
                    attention_weights = outputs[1][self.sv_layer_pos][:, i, 0, :]
                    loss_att += self.att_lambda * masked_cross_entropy(
                        attention_weights, attention_vals, attention_mask
                    )
                loss = loss + loss_att
            outputs = (loss,) + outputs

        # (loss), logits, (hidden_states), (attentions)
        return outputs
