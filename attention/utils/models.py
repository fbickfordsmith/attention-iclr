import tensorflow as tf

def attention_network(attention_layer, position='block5_pool'):
    """
    Build an attention network by taking a pretrained VGG16 and inserting an
    attention layer. Fix all weights except for the attention weights.
    """
    vgg = tf.keras.applications.VGG16()
    model = tf.keras.models.Sequential()
    for layer in vgg.layers:
        layer.trainable = False
        model.add(layer)
        if layer.name == position:
            model.add(attention_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=3e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model
