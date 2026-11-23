import 'dart:convert';
import 'dart:io';
import 'package:expense_mobile/arabic_features.dart';

Future<void> main() async {
  final encoder = ArabicFeatures.fromJson(await File('assets/arabic_features.json').readAsString());
  await for (final line in stdin.transform(utf8.decoder).transform(const LineSplitter())) {
    final vector = encoder.encode(jsonDecode(line) as String);
    stdout.writeln(jsonEncode({for (var i = 0; i < vector.length; i++)
      if (vector[i] != 0) i.toString(): vector[i]}));
  }
}
