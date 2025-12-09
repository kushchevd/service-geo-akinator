import onnx
import onnxruntime
from transformers import BertTokenizer


class GeoAkinatorModelOnnx:
    def __init__(self, onnx_model_path: str, tokenizer_path: str):
        self.model_path = onnx_model_path
        self.session = onnxruntime.InferenceSession(self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.metadata = None

    def predict(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        fmt_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        output = self.session.run(None, fmt_inputs)
        lat, lon = output[0][0].tolist()
        return lat, lon

    def get_metadata(self):
        try:
            if self.metadata is None:
                model = onnx.load(self.model_path)
                metadata = {}
                if model.metadata_props:
                    for prop in model.metadata_props:
                        metadata[prop.key] = prop.value
                self.metadata = metadata
            return self.metadata

        except Exception as e:
            raise Exception(f"Failed to load metadata: {e}")
