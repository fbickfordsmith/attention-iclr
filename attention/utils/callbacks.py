import warnings
import numpy as np
import tensorflow as tf

class RelativeEarlyStopping(tf.keras.callbacks.Callback):
    """
    Stop training when a monitored quantity has stopped improving. Difference
    compared to `tf.keras.callbacks.EarlyStopping`: here, the `min_delta`
    parameter defines a relative change, not an absolute one. See TensorFlow
    docs [1] for more detail. Implemented by Ken Luo [2].

    Example:
    ```python
    # This callback stops training when the validation-set loss does not improve
    # by at least 1% (relative to previous epoch) for two consecutive epochs.
    earlystopping = RelativeEarlyStopping(min_delta=0.01, patience=2)
    ```

    References:
    [1] tensorflow.org/api_docs/python/tf/keras/callbacks
    [2] github.com/don-tpanic
    """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(RelativeEarlyStopping, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        # DEBUGGING
        print('\n\n')
        relative_change = (abs(self.best - current) / self.best) * 100
        print('----------------------------')
        print('relative change = {%.3f} percent' % relative_change)
        print('----------------------------')
        print('best = %s' % self.best)
        print('current = %s' % current)
        print('----------------------------')
        print('\n\n')

        # Line below is the only change vs `tf.keras.callbacks.EarlyStopping`.
        if self.monitor_op(current, self.best * (1 - self.min_delta)):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value