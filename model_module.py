import torch
from torch import nn
from transformers import AutoModel, EvalPrediction


class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self._init_weights()

    def forward(self, sequence_output, pooled_output, labels=None, **kwargs):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if labels.dim() != 1:
                # Remove padding
                labels = labels[:, 0]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.long().view(-1)
            )

        return logits, loss

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()


class MultiTaskModel(nn.Module):
    def __init__(self, encoder_name_or_path, num_labels, aspect_ids: List):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)

        self.output_heads = nn.ModuleDict()
        for aspect_idx in aspect_ids:
            decoder = SequenceClassificationHead(self.encoder.config.hidden_size, num_labels)
            # ModuleDict requires keys to be strings
            self.output_heads[str(aspect_idx)] = decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        aspect_ids=None,
        **kwargs,
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]

        unique_aspect_ids_list = torch.unique(aspect_ids).tolist()

        loss_list = []
        logits = None
        for unique_aspect_id in unique_aspect_ids_list:

            aspect_id_filter = aspect_ids == unique_aspect_id
            logits, aspect_loss = self.output_heads[str(unique_aspect_id)].forward(
                sequence_output[aspect_id_filter],
                pooled_output[aspect_id_filter],
                labels=None if labels is None else labels[aspect_id_filter],
                attention_mask=attention_mask[aspect_id_filter],
            )

            if labels is not None:
                loss_list.append(aspect_loss)

        # logits are only used for eval. and in case of eval the batch is not multi task
        # For training only the loss is used
        outputs = (logits, outputs[2:])

        if loss_list:
            loss = torch.stack(loss_list)
            outputs = (loss.mean(),) + outputs

        return outputs


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}