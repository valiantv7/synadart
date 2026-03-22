import 'dart:math';
import 'package:test/test.dart';
import 'package:synadart/synadart.dart';

void main() {
  group('Softmax Activation', () {
    test('Softmax output sums to 1', () {
      final layer = Layer(size: 5, activation: ActivationAlgorithm.softmax);
      layer.initialise(parentLayerSize: 3, learningRate: 0.1);

      // Provide some inputs
      layer.accept([1.0, 2.0, 3.0]);

      final outputs = layer.output;
      expect(outputs.length, equals(5));

      double sum = outputs.reduce((a, b) => a + b);
      expect(sum, closeTo(1.0, 1e-9));

      for (var out in outputs) {
        expect(out, greaterThan(0));
        expect(out, lessThan(1));
      }
    });

    test('Softmax correctly identifies the largest input', () {
      // With very different weighted sums, the largest one should have the highest probability
      // We'll manually set weights to make it predictable if possible,
      // but easier is just to check if it's monotonic with respect to exp(weightedSum)

      final layer = Layer(size: 3, activation: ActivationAlgorithm.softmax);
      layer.initialise(parentLayerSize: 1, learningRate: 0.1);

      // Manually set weights for predictability
      layer.neurons[0].weights = [1.0]; // weighted sum = input
      layer.neurons[1].weights = [2.0]; // weighted sum = 2 * input
      layer.neurons[2].weights = [3.0]; // weighted sum = 3 * input

      layer.accept([1.0]); // weighted sums: 1, 2, 3

      final outputs = layer.output;
      // exp(1) ~ 2.718
      // exp(2) ~ 7.389
      // exp(3) ~ 20.085
      // sum ~ 30.192
      // expected: [0.09, 0.24, 0.66]

      expect(outputs[2], greaterThan(outputs[1]));
      expect(outputs[1], greaterThan(outputs[0]));

      double total = exp(1.0) + exp(2.0) + exp(3.0);
      expect(outputs[0], closeTo(exp(1.0) / total, 1e-9));
      expect(outputs[1], closeTo(exp(2.0) / total, 1e-9));
      expect(outputs[2], closeTo(exp(3.0) / total, 1e-9));
    });

    test('Sequential network with Softmax', () {
      final network = Sequential(
        learningRate: 0.1,
        layers: [
          Layer(size: 4, activation: ActivationAlgorithm.relu),
          Layer(size: 3, activation: ActivationAlgorithm.softmax),
        ],
      );

      final output = network.process([1.0, 0.5, -0.2, 0.8]);
      expect(output.length, equals(3));
      expect(output.reduce((a, b) => a + b), closeTo(1.0, 1e-9));
    });
  });
}
