"""Rebuild Arabic mobile assets: install ai-edge-litert==2.4.0 in the project venv."""
import json
from pathlib import Path
import sys

from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter
import flatbuffers
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def scoring_graph(weights, bias):
    """A linear layer, seven sigmoid calibrations, then row normalization."""
    model = schema.ModelT()
    model.version = 3
    model.operatorCodes = []
    model.buffers = [schema.BufferT()]
    graph = schema.SubGraphT()
    graph.name = "Arabic expense classifier"
    graph.tensors = []
    graph.operators = []
    model.subgraphs = [graph]

    def tensor(name, shape, value=None, dtype=schema.TensorType.FLOAT32):
        t = schema.TensorT()
        t.name, t.shape, t.type = name, shape, dtype
        if value is not None:
            buf = schema.BufferT()
            buf.data = np.frombuffer(value.tobytes(), dtype=np.uint8)
            t.buffer = len(model.buffers)
            model.buffers.append(buf)
        graph.tensors.append(t)
        return len(graph.tensors) - 1

    def op(code, inputs, output, options=None):
        oc = schema.OperatorCodeT()
        oc.builtinCode = oc.deprecatedBuiltinCode = code
        operator = schema.OperatorT()
        operator.opcodeIndex = len(model.operatorCodes)
        model.operatorCodes.append(oc)
        operator.inputs, operator.outputs = inputs, [output]
        if options is not None:
            operator.builtinOptions = options
            operator.builtinOptionsType = getattr(schema.BuiltinOptions, type(options).__name__[:-1])
        graph.operators.append(operator)

    classes, size = weights.shape
    x = tensor("tfidf", [1, size])
    w = tensor("weights", [classes, size], weights.astype("<f4"))
    b = tensor("bias", [classes], bias.astype("<f4"))
    logits = tensor("calibrated_logits", [1, classes])
    scores = tensor("sigmoid_scores", [1, classes])
    axis = tensor("class_axis", [1], np.array([1], dtype="<i4"), schema.TensorType.INT32)
    total = tensor("sum", [1, 1])
    probs = tensor("probabilities", [1, classes])
    codes = schema.BuiltinOperator
    op(codes.FULLY_CONNECTED, [x, w, b], logits, schema.FullyConnectedOptionsT())
    op(codes.LOGISTIC, [logits], scores)
    reducer = schema.ReducerOptionsT()
    reducer.keepDims = True
    op(codes.SUM, [scores, axis], total, reducer)
    op(codes.DIV, [scores, total], probs, schema.DivOptionsT())
    graph.inputs, graph.outputs = [x], [probs]
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def main():
    model = joblib.load(ROOT / "models/arabic.joblib")
    member, = model.calibrated_classifiers_
    features, svm = member.estimator.steps[0][1], member.estimator.steps[1][1]
    slopes = np.array([-c.a_ for c in member.calibrators])
    offsets = np.array([-c.b_ for c in member.calibrators])
    binary = scoring_graph(slopes[:, None] * svm.coef_, slopes * svm.intercept_ + offsets)
    metadata = {"classes": model.classes_.tolist(), "blocks": []}
    for name, vec in features.transformer_list:
        metadata["blocks"].append({
            "name": name, "analyzer": vec.analyzer, "range": vec.ngram_range,
            "terms": vec.get_feature_names_out().tolist(), "idf": vec.idf_.tolist(),
            "norm": vec.norm, "weight": features.transformer_weights[name],
        })

    runtime = Interpreter(model_content=binary)
    runtime.allocate_tensors()
    source = runtime.get_input_details()[0]["index"]
    dest = runtime.get_output_details()[0]["index"]
    rows = pd.read_csv(ROOT / "data/arabic_expenses.csv").description.tolist()
    rows.append("دفعت فاتورة الإنترنت")
    vectors = features.transform(rows)
    expected = model.predict_proba(rows)
    worst = 0
    for i, reference in enumerate(expected):
        runtime.set_tensor(source, vectors[i].toarray().astype(np.float32))
        runtime.invoke()
        actual = runtime.get_tensor(dest)[0]
        np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=0)
        assert actual.argmax() == reference.argmax()
        worst = max(worst, float(abs(actual-reference).max()))
    folder = ROOT / "mobile/assets"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "arabic.tflite").write_bytes(binary)
    (folder / "arabic_features.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf8")
    print(f"{len(rows)} matching labels; maximum probability difference {worst:.9f}")
    print(rows[-1], model.classes_[actual.argmax()], actual.max())


if __name__ == "__main__":
    main()
