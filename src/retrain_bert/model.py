import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import lightning as L

from retrain_bert import settings
from retrain_bert.metrics import cumulative_accuracy


def add_gaussian_noise(tensor, mean=0.0, std=0.1):
    return tensor + torch.randn(tensor.size(), device=tensor.device) * std + mean


class AlgoriAecocCategorization(L.LightningModule):
    def __init__(
        self,
        labels_conf,
        num_levels=settings.DEEPEST_LEVEL,
        lr=1e-3,
        max_sequence_length=settings.MAX_SEQUENCE_LENGTH,
    ):
        super(AlgoriAecocCategorization, self).__init__()
        self.num_levels = num_levels
        self.labels_conf = labels_conf
        self.lr = lr
        self.bert_model = BertModel.from_pretrained("bert-base-multilingual-uncased")

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(768, 16, kernel_size=3, padding=1),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
            ]
        )

        self.dense_layers = nn.ModuleList(
            [
                nn.Linear(768, self.num_levels**2),
                nn.Linear(self.num_levels**2, self.num_levels**2),
                nn.Linear(self.num_levels**2, self.num_levels**2),
                nn.Linear(self.num_levels**2, self.num_levels**2),
            ]
        )

        first_head = nn.Linear(
            self.conv_layers[-1].out_channels * max_sequence_length
            + self.dense_layers[-1].out_features,
            labels_conf[0]["num_classes"],
        )

        self.output_heads = nn.ModuleList(
            [first_head]
            + [
                nn.Linear(
                    first_head.in_features
                    + first_head.out_features
                    + sum([labels_conf[l]["num_classes"] for l in range(1, level)]),
                    labels_conf[level]["num_classes"],
                )
                for level in range(1, self.num_levels)
            ]
        )

        self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.bert_model.parameters():
            param.requires_grad = True

    def fine_tune(self):
        self.unfreeze_backbone()
        self.lr = self.lr / 100

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooled_output = encoder_outputs.pooler_output
        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = sequence_output.permute(0, 2, 1)

        if self.training:
            pooled_output = add_gaussian_noise(pooled_output)
            sequence_output = add_gaussian_noise(sequence_output)
            sequence_output = self.dropout(sequence_output)

        x = pooled_output
        for dense_layer in self.dense_layers:
            x = F.relu(dense_layer(x))

        for conv_layer in self.conv_layers:
            sequence_output = F.relu(conv_layer(sequence_output))
        sequence_output = self.flatten(sequence_output)

        x = torch.cat((x, sequence_output), dim=1)

        if self.training:
            x = self.dropout(x)

        heads = []
        for head in self.output_heads:
            head_output = head(x)
            x = torch.cat((x, head_output), dim=1)
            heads.append(head_output)

        return heads

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self.forward(input_ids, attention_mask)
        loss = 0
        for i, head_output in enumerate(outputs):
            loss += F.cross_entropy(head_output, labels[i])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self.forward(input_ids, attention_mask)
        loss = 0
        for i, head_output in enumerate(outputs):
            loss += F.cross_entropy(head_output, labels[i])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        accuracy = cumulative_accuracy(labels, outputs)
        for i, acc in enumerate(accuracy):
            self.log(f"val_accuracy_level_{i + 1}", acc, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
