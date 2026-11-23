import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from src.text_processing import clean_text


class Encoder:
    def __init__(self, folder):
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<pad>"), pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        path = folder / "model_quantized.onnx"
        self.session = ort.InferenceSession(str(path if path.exists() else folder / "onnx" / path.name),
                                            options, providers=["CPUExecutionProvider"])

    def encode(self, descriptions):
        keys = [clean_text(x) for x in descriptions]
        cache = {}
        missing = sorted(set(keys), key=lambda k: (len(k), k))
        for pos in range(0, len(missing), 32):
            batch = missing[pos:pos+32]
            tokens = self.tokenizer.encode_batch(batch)
            ids = np.asarray([t.ids for t in tokens], dtype=np.int64)
            mask = np.asarray([t.attention_mask for t in tokens], dtype=np.int64)
            inputs = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            hidden = self.session.run(None, {i.name: inputs[i.name] for i in self.session.get_inputs()})[0]
            pooled = (hidden * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1)[:, None]
            vectors = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            cache.update(zip(batch, vectors.astype(np.float32)))
        return np.stack([cache[k] for k in keys])
