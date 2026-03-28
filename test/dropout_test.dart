import 'package:synadart/synadart.dart';
import 'package:test/test.dart';

void main() {
  group('Dropout Tests', () {
    test('Dropout zeros some outputs during training', () {
      final network = Sequential(
        learningRate: 0.1,
        layers: [
          Dense(size: 100, activation: ActivationAlgorithm.relu, dropoutRate: 0.5),
          Dense(size: 1, activation: ActivationAlgorithm.sigmoid),
        ],
      );

      // During inference, dropout should NOT be active
      network.isTraining = false;
      final inputs = List.generate(100, (index) => 1.0);
      final outputInference = network.process(inputs);
      expect(outputInference.every((e) => e > 0), isTrue);

      // During training, dropout should be active
      network.isTraining = true;
      final outputTraining = network.process(inputs);
      // Since size is 100 and rate is 0.5, some should be 0.
      // But we're looking at the final output of the network, which is only 1 neuron.
      // We need to check the layer's output.
      final hiddenLayerOutput = network.layers[0].output;
      expect(hiddenLayerOutput.any((e) => e == 0), isTrue);
    });

    test('Dropout scales outputs during training', () {
      final layer = Dense(size: 10, activation: ActivationAlgorithm.relu, dropoutRate: 0.5);
      layer.initialise(parentLayerSize: 10, learningRate: 0.1);

      // Fixed inputs for deterministic test (as much as possible)
      final inputs = List.generate(10, (index) => 1.0);
      layer.accept(inputs);

      // Inference
      layer.isTraining = false;
      final outputInference = List.from(layer.output);

      // Training
      layer.isTraining = true;
      // We can't be sure which ones are dropped, but the ones that AREN'T dropped should be scaled by 1/(1-0.5) = 2
      final outputTraining = layer.output;

      for (var i = 0; i < 10; i++) {
        if (outputTraining[i] != 0) {
          expect(outputTraining[i], closeTo(outputInference[i] * 2, 0.0001));
        }
      }
    });

    test('Dropout mask is applied to gradients during backpropagation', () {
      // Create a network with dropout
      final network = Sequential(
        learningRate: 0.1,
        layers: [
          Dense(size: 1, activation: ActivationAlgorithm.relu, dropoutRate: 0.999), // Very likely to drop
          Dense(size: 1, activation: ActivationAlgorithm.sigmoid),
        ],
      );

      final input = [1.0];
      final expected = [0.0];

      final weightsBefore = List.from(network.layers[0].neurons[0].weights);
      network.train(inputs: [input], expected: [expected], iterations: 1, quiet: true);
      final weightsAfter = List.from(network.layers[0].neurons[0].weights);
      expect(weightsBefore, equals(weightsAfter)); // Since it's dropped with 0.999 prob
    });

    test('LSTM Dropout zeros some outputs during training', () {
      final network = Sequential(
        learningRate: 0.1,
        layers: [
          LSTM(
              size: 10,
              activation: ActivationAlgorithm.relu,
              recurrenceActivation: ActivationAlgorithm.sigmoid,
              dropoutRate: 0.5),
          Dense(size: 1, activation: ActivationAlgorithm.sigmoid),
        ],
      );

      network.isTraining = true;
      final inputs = List.generate(10, (index) => 1.0);
      network.process(inputs);

      final lstmLayerOutput = network.layers[0].output;
      expect(lstmLayerOutput.any((e) => e == 0), isTrue);
    });
  });
}
