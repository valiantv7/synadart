import 'package:sprint/sprint.dart';

import 'package:synadart/src/layers/layer.dart';

/// Representation of a neural network containing `Layers`, which each further
/// house a number of `Neurons`.  A `Network` takes an input in the form of
/// several entries and returns an output by processing the data by running the
/// data through the layers.
///
/// In order to train a `Network`, the selected `Network` must have a training
/// algorithm mixed into it - most commonly `Backpropagation`.
class Network {
  /// Used for performance analysis as well as general information logging.
  final Stopwatch stopwatch = Stopwatch();

  /// `Sprint` instance for logging messages.
  final Sprint log = Sprint('Network');

  /// The `Layers` part of this `Network`.
  final List<Layer> layers = [];

  /// The degree of radicality at which the `Network` will adjust its `Neurons`
  /// weights.
  double learningRate;

  /// The maximum value that the weight margin can take during training.
  /// If set to 0, no clipping will be performed.
  double gradientClipping;

  /// Whether or not this `Network` is currently in training mode.
  bool isTraining = false;

  /// Creates a `Network` with optional `Layers`.
  ///
  /// [learningRate] - The level of aggressiveness at which this `Network` will
  /// adjust its `Neurons`' weights during training.
  ///
  /// [layers] - (Optional) The `Layers` of this `Network`.
  ///
  /// [gradientClipping] - (Optional) The maximum value that the weight margin
  /// can take during training.
  Network({
    required this.learningRate,
    List<Layer>? layers,
    this.gradientClipping = 0,
  }) {
    if (layers != null) {
      addLayers(layers);
    }
  }

  /// Processes the [inputs] by propagating them across every `Layer`.
  /// and returns the output.
  List<double> process(List<double> inputs) {
    var output = inputs;

    for (final layer in layers) {
      layer.isTraining = isTraining;
      layer.accept(output);
      output = layer.output;
    }

    return output;
  }

  /// Adds a `Layer` to this `Network`.
  void addLayer(Layer layer) {
    layer.initialise(
      parentLayerSize: layers.isEmpty ? 0 : layers.last.size,
      learningRate: learningRate,
      gradientClipping: gradientClipping,
    );

    layers.add(layer);

    log.info('Added layer of size ${layer.neurons.length}.');
  }

  /// Adds a list of `Layers` to this `Network`.
  void addLayers(List<Layer> layers) {
    for (final layer in layers) {
      addLayer(layer);
    }
  }

  /// Clears the `Network` by removing all `Layers`, thereby returning it to its
  /// initial, empty state.
  void clear() {
    if (layers.isEmpty) {
      log.warning('Attempted to reset an already empty network.');
      return;
    }

    stopwatch.reset();
    layers.clear();

    log.success('Network reset successfully.');
  }
}
