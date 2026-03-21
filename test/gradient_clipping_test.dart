import 'package:synadart/synadart.dart';
import 'package:test/test.dart';

void main() {
  test('Gradient clipping prevents explosion', () {
    // Use the same parameters that caused explosion before, but with clipping
    final network = Sequential(
      learningRate: 100.0,
      gradientClipping: 1.0, // Enable clipping
      layers: [
        Layer(size: 2, activation: ActivationAlgorithm.relu),
        Layer(size: 1, activation: ActivationAlgorithm.relu),
      ],
    );

    final inputs = List.generate(10, (index) => [1e5, 1e5]);
    final expected = List.generate(10, (index) => [0.0]);

    // Train with clipping
    network.train(inputs: inputs, expected: expected, iterations: 1000, quiet: true);

    // Check if weights are NOT NaN and not huge
    bool exploded = false;
    for (var layer in network.layers) {
      for (var neuron in layer.neurons) {
        for (var weight in neuron.weights) {
          if (weight.isNaN || weight.isInfinite || weight.abs() > 1e10) {
            exploded = true;
            break;
          }
        }
        if (exploded) break;
      }
      if (exploded) break;
    }

    if (exploded) {
      print('Explosion detected despite clipping!');
    } else {
      print('No explosion detected with clipping.');
    }

    expect(exploded, isFalse, reason: 'Gradient should NOT have exploded with clipping');
  });
}
