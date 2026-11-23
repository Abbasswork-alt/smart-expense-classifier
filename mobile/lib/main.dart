import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

import 'arabic_classifier.dart';

void main() => runApp(const ExpenseApp());

const _categoryNames = {
  'Bills': 'فواتير',
  'Entertainment': 'ترفيه',
  'Food': 'طعام',
  'Health': 'صحة',
  'Other': 'أخرى',
  'Shopping': 'تسوّق',
  'Transport': 'مواصلات',
};

class ExpenseApp extends StatelessWidget {
  const ExpenseApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'تصنيف المصروفات',
        locale: const Locale('ar'),
        supportedLocales: const [Locale('ar')],
        localizationsDelegates: GlobalMaterialLocalizations.delegates,
        theme: ThemeData(colorSchemeSeed: const Color(0xff14685a)),
        home: const _ExpenseScreen(),
      );
}

class _ExpenseScreen extends StatefulWidget {
  const _ExpenseScreen();

  @override
  State<_ExpenseScreen> createState() => _ExpenseScreenState();
}

class _ExpenseScreenState extends State<_ExpenseScreen> {
  final _description = TextEditingController();
  ArabicClassifier? _classifier;
  Prediction? _result;
  String? _message;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final classifier = await ArabicClassifier.load();
      if (!mounted) {
        classifier.close();
        return;
      }
      setState(() => _classifier = classifier);
    } catch (error, stack) {
      debugPrintStack(
          label: 'Arabic model loading failed: $error', stackTrace: stack);
      if (mounted) setState(() => _message = 'تعذّر تحميل نموذج التصنيف.');
    }
  }

  void _classify() {
    final result = _classifier!.classify(_description.text);
    setState(() {
      _result = result;
      _message = result == null ? 'أدخل وصفًا أوضح للمصروف.' : null;
    });
  }

  @override
  void dispose() {
    _description.dispose();
    _classifier?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final result = _result;
    return Scaffold(
      appBar: AppBar(title: const Text('تصنيف المصروفات')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(24),
          children: [
            TextField(
              controller: _description,
              enabled: _classifier != null,
              decoration: const InputDecoration(
                labelText: 'وصف المصروف',
                hintText: 'مثلاً: دفعت فاتورة الإنترنت',
                border: OutlineInputBorder(),
              ),
              onChanged: (_) {
                if (_result != null || _message != null) {
                  setState(() {
                    _result = null;
                    _message = null;
                  });
                }
              },
              onSubmitted: (_) => _classify(),
            ),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: _classifier == null ? null : _classify,
              child: const Text('صنّف المصروف'),
            ),
            const SizedBox(height: 20),
            if (_classifier == null && _message == null)
              const LinearProgressIndicator(),
            if (_message != null) Text(_message!),
            if (result != null) ...[
              Text(_categoryNames[result.category] ?? result.category,
                  style: Theme.of(context).textTheme.headlineSmall),
              Text(
                  'ثقة النموذج: ${(100 * result.probability).toStringAsFixed(1)}%'),
            ],
          ],
        ),
      ),
    );
  }
}
