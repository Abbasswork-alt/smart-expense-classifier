import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:unorm_dart/unorm_dart.dart' as unicode;

class ArabicFeatures {
  final List<_Block> _blocks;
  final List<String> classes;
  final int size;

  ArabicFeatures._(this._blocks, this.classes)
      : size = _blocks.fold(0, (n, b) => n + b.idf.length);

  factory ArabicFeatures.fromJson(String asset) {
    final data = jsonDecode(asset) as Map;
    return ArabicFeatures._(
      [for (final item in data['blocks']) _Block(item)],
      List<String>.from(data['classes']),
    );
  }

  static final _marks = RegExp(r'[\u0640\u064b-\u0652\u0670]');
  static final _letters = RegExp(r'\p{L}+', unicode: true);

  Float32List encode(String text) {
    final clean = unicode.nfc(text.toLowerCase()).replaceAll(_marks, '');
    final words = _letters.allMatches(clean).map((m) => m[0]!).toList();
    final output = Float32List(size);
    var offset = 0;
    for (final block in _blocks) {
      block.write(words, output, offset);
      offset += block.idf.length;
    }
    return output;
  }
}

class _Block {
  final Map<String, int> index;
  final List<double> idf;
  final String analyzer;
  final int first, last;
  final double weight;

  _Block(Map data)
      : index = List<String>.from(data['terms']).asMap().map((i, s) => MapEntry(s, i)),
        idf = List<double>.from(data['idf']),
        analyzer = data['analyzer'],
        first = data['range'][0],
        last = data['range'][1],
        weight = (data['weight'] as num).toDouble() {
    if (data['norm'] != 'l2' || !{'word', 'char_wb'}.contains(analyzer)) {
      throw FormatException('Unsupported feature configuration');
    }
  }

  Iterable<String> terms(List<String> words) sync* {
    if (analyzer == 'word') {
      final tokens = words.where((w) => w.runes.length >= 2).toList();
      for (var n = first; n <= last; n++) {
        for (var i = 0; i + n <= tokens.length; i++) {
          yield tokens.sublist(i, i + n).join(' ');
        }
      }
      return;
    }
    for (final word in words) {
      final padded = ' $word '.runes.toList();
      for (var n = first; n <= last; n++) {
        if (n >= padded.length) {
          yield String.fromCharCodes(padded);
          break;
        }
        for (var i = 0; i + n <= padded.length; i++) {
          yield String.fromCharCodes(padded, i, i + n);
        }
      }
    }
  }

  void write(List<String> words, Float32List output, int offset) {
    final counts = <int, int>{};
    for (final term in terms(words)) {
      final i = index[term];
      if (i != null) counts[i] = (counts[i] ?? 0) + 1;
    }
    var sumSquares = 0.0;
    for (final e in counts.entries) {
      final value = e.value * idf[e.key];
      sumSquares += value * value;
    }
    if (sumSquares == 0) return;
    final scale = weight / math.sqrt(sumSquares);
    for (final e in counts.entries) {
      output[offset + e.key] = e.value * idf[e.key] * scale;
    }
  }
}
