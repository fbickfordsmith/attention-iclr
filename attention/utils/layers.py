import tensorflow as tf

class ElementwiseAttention(tf.keras.layers.Layer):
    """
    Multiply each element of an input tensor by a separate attention weight.

    References:
    - stackoverflow.com/questions/46821845/how-to-add-a-trainable-hadamard-product-layer-in-keras
    - keras.io/layers/writing-your-own-keras-layers
    - tensorflow.org/guide/keras/custom_layers_and_models
    """
    def __init__(self, **kwargs):
        super(ElementwiseAttention, self).__init__(name='attention', **kwargs)

    def build(self, input_shape):
        super(ElementwiseAttention, self).build(input_shape)
        self.attention_weights = self.add_weight(
            name='attention_weights',
            shape=(1,)+input_shape[1:],
            initializer='ones',
            constraint=tf.keras.constraints.NonNeg())

    def call(self, x):
        return x * self.attention_weights

    def compute_output_shape(self, input_shape):
        return input_shape
