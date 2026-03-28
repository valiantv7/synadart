import 'package:synadart/src/layers/layer.dart';
import 'package:synadart/src/networks/network.dart';
import 'package:synadart/src/networks/training/backpropagation.dart';
import 'package:synadart/src/utils/utils.dart';

/// A `Network` model in which every `Layer` has one input and one output
/// tensor.
class Sequential extends Network with Backpropagation {
  final String _layersField = 'layers';
  final String _learningRateField = 'learningRate';
  final String _gradientClippingField = 'gradientClipping';

  /// Creates a `Sequential` model network.
  ///
  /// [learningRate] - The level of aggressiveness at which this `Network` will
  /// adjust its `Neurons`' weights during training.
  ///
  /// [layers] - (Optional) The `Layers` of this `Network`.
  ///
  /// [gradientClipping] - (Optional) The maximum value that the weight margin
  /// can take during training.
  Sequential({
    required super.learningRate,
    super.layers,
    super.gradientClipping,
  });

  /// Loads a model from a JSON .
  Sequential.fromMap(Map<String, dynamic> data) : super(learningRate: 0) {
    learningRate = data[_learningRateField];
    gradientClipping = (data[_gradientClippingField] as double? ?? 0.0);
    for (Map<String, dynamic> layer in data[_layersField]) {
      layers.add(Layer.fromJson(layer));
    }
  }

  /// Save the model to a JSON.
  Map<String, dynamic> toMap() {
    return {
      _layersField: layers.map((e) => e.toJson()).toList(),
      _learningRateField: learningRate,
      _gradientClippingField: gradientClipping ?? 0.0
    };
  }

  Sequential variation({Mutation? mutation}) {
    return Sequential(
      learningRate: learningRate,
    )..layers.addAll(layers.map((e) {
        return e.isInput ? e.copyWith() : e.variation(mutation: mutation);
      }).toList());
  }
}
