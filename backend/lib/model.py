import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertPreTrainedModel, BertTokenizer


class BoundedLatLonRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        latlon = self.mlp(x)
        lat = latlon[:, 0] * 90
        lon = latlon[:, 1] * 180
        return torch.stack([lat, lon], dim=1)


class BertForLatLonCosineLoss(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.regressor = BoundedLatLonRegressor(config.hidden_size)
        self.init_weights()

    def latlon_to_xyz(self, latlon):
        lat_rad = torch.deg2rad(latlon[:, 0])
        lon_rad = torch.deg2rad(latlon[:, 1])
        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.cos(lat_rad) * torch.sin(lon_rad)
        z = torch.sin(lat_rad)
        return torch.stack([x, y, z], dim=1)

    def predict(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        latlon_pred = self.regressor(pooled_output)
        return latlon_pred

    def forward(self, input_ids, attention_mask, labels=None):
        latlon_pred = self.predict(input_ids, attention_mask)

        loss = None
        if labels is not None:
            pred_xyz = self.latlon_to_xyz(latlon_pred)
            true_xyz = self.latlon_to_xyz(labels)
            cosine_loss = 1 - (pred_xyz * true_xyz).sum(dim=1)
            loss = cosine_loss.mean()

        return {"loss": loss, "logits": latlon_pred}


class BertForLatLonInference(BertForLatLonCosineLoss):
    def forward(self, input_ids, attention_mask):
        return self.predict(input_ids, attention_mask)


class GeoAkinatorModel:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        config_path: str = None,
        exp_name: str = None,
    ):
        self.model, self.tokenizer = self._load_model(
            model_path, tokenizer_path, config_path
        )
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.config_path = config_path
        self.exp_name = exp_name

    def _load_model(
        self, model_path: str, tokenizer_path: str, config_path: str = None
    ):
        if config_path is not None:
            self.config = BertConfig.from_pretrained(config_path)
        else:
            self.config = None
        model = BertForLatLonCosineLoss.from_pretrained(model_path, config=self.config)
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        return model.eval(), tokenizer

    def predict(self, x):
        with torch.no_grad():
            inputs = self.tokenizer(
                x,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128,
            )
            output = self.model.predict(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            lat, lon = output[0].tolist()
        return lat, lon
