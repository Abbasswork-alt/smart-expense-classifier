import 'package:expense_mobile/arabic_classifier.dart';
import 'package:expense_mobile/main.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  const bill = 'دفعت فاتورة الإنترنت';

  testWidgets('Arabic assets predict the Python reference on device',
      (tester) async {
    final classifier = await ArabicClassifier.load();
    addTearDown(classifier.close);
    final result = classifier.classify(bill)!;
    expect(result.category, 'Bills');
    expect(result.probability, closeTo(0.9920125927253274, 1e-5));
    debugPrint('Android TFLite: ${result.category} ${result.probability}');
  });

  testWidgets('Arabic form classifies repeated and vocalized expenses offline',
      (tester) async {
    await tester.pumpWidget(const ExpenseApp());
    await tester.pumpAndSettle();
    final field = find.byType(TextField);
    final button = find.widgetWithText(FilledButton, 'صنّف المصروف');

    Future<void> submit(String text) async {
      await tester.enterText(field, text);
      await tester.tap(button);
      await tester.pumpAndSettle();
    }

    for (final text in [bill, 'دَفَعْتُ فَاتُورَةَ الإِنْتَرْنِتِ']) {
      await submit(text);
      expect(find.text('فواتير'), findsOneWidget);
      expect(find.text('ثقة النموذج: 99.2%'), findsOneWidget);
    }
    await submit('اشتريت دواء من الصيدلية');
    expect(find.text('صحة'), findsOneWidget);
    for (final text in ['   ', '!!!']) {
      await submit(text);
      expect(find.text('أدخل وصفًا أوضح للمصروف.'), findsOneWidget);
      expect(find.text('صحة'), findsNothing);
    }
    await submit(bill);
    expect(find.text('فواتير'), findsOneWidget);
    expect(find.text('ثقة النموذج: 99.2%'), findsOneWidget);
  });
}
