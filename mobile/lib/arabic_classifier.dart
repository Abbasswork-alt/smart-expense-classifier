import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'arabic_features.dart';

typedef Prediction = ({String category, double probability});

class ArabicClassifier {
  final ArabicFeatures _features;
  final Interpreter _interpreter;

  ArabicClassifier._(this._features, this._interpreter);

  static Future<ArabicClassifier> load() async {
    final features = ArabicFeatures.fromJson(
      await rootBundle.loadString('assets/arabic_features.json'),
    );
    final interpreter = await Interpreter.fromAsset('assets/arabic.tflite');
    try {
      final input = interpreter.getInputTensors().single;
      final output = interpreter.getOutputTensors().single;
      if (input.type != TensorType.float32 ||
          output.type != TensorType.float32 ||
          !listEquals(input.shape, [1, features.size]) ||
          !listEquals(output.shape, [1, features.classes.length])) {
        throw FormatException(
            'Arabic feature metadata and model tensors disagree.');
      }
      return ArabicClassifier._(features, interpreter);
    } catch (_) {
      interpreter.close();
      rethrow;
    }
  }

  Prediction? classify(String description) {
    if (description.trim().isEmpty) return null;
    final row = _features.encode(description);
    if (!row.any((value) => value != 0)) return null;

    final scores = Float32List(_features.classes.length);
    // The model already has a fixed batch size of one.
    _interpreter.run(row.buffer, scores.buffer);
    var best = 0;
    for (var i = 1; i < scores.length; i++) {
      if (scores[i] > scores[best]) best = i;
    }
    return (category: _features.classes[best], probability: scores[best]);
  }

  void close() => _interpreter.close();
}
