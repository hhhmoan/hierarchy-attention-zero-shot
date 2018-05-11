import tensorflow as tf
import utils
from tensorflow.python import debug as tf_debug
class Model:
    def __init__(self, class_num=10, lr=0.001, decay_steps=1000, decay_rate=0.5, train_log_print_iter=10):
        self.classifier = tf.estimator.Estimator(model_fn=self.build_model, model_dir='./model/', params={
            'lr': lr,
            'decay_steps': decay_steps,
            'decay_rate': decay_rate,
            'class_num': class_num
        })
        self.train_hooks = [tf.train.LoggingTensorHook(tensors=self._train_tensor_log(),
                                                       every_n_iter=train_log_print_iter)]
        #self.train_hooks.append(tf_debug.LocalCLIDebugHook())

    @staticmethod
    def _train_tensor_log():
        _TENSOR = ['learning_rate', 'losses', 'accuracy', 'global_step']
        return dict((x,x) for x in _TENSOR)

    def build_model(self, features, labels, mode, params):
        class_num = params['class_num']
        lr = params['lr']
        decay_steps = params['decay_steps']
        decay_rate = params['decay_rate']
        logits = self.forward(features, mode == tf.estimator.ModeKeys.TRAIN)
        predicted_classes = tf.argmax(logits, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'probabilities': tf.nn.softmax(logits),
                'logits': logits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predicted_classes,
                                       name='acc_op')
        metrics = {'accuracy': accuracy}
        tf.identity(accuracy[1], 'accuracy')
        tf.summary.scalar('accuracy', accuracy[1])
        loss = self.loss(labels, logits)
        tf.identity(loss, name='losses')
        tf.summary.scalar('losses', loss)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)
        assert mode == tf.estimator.ModeKeys.TRAIN

        learning_rate = tf.train.exponential_decay(learning_rate=lr, global_step=tf.train.get_global_step(),
                                                   decay_steps=decay_steps, decay_rate=decay_rate)
        tf.identity(learning_rate, 'learning_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        tf.identity(tf.train.get_global_step(), 'global_step')
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    @staticmethod
    def forward(features, training=False):
        F = features['feature']
        A = features['att_vec']
        K = features['knowledge']
        mask = features['mask']
        s = utils.attention_vector(A, F.shape[2])
        feature = utils.attention_layer(F, s)
        match = utils.transform2knowledge(feature, K.shape[-1])
        match = tf.tile(tf.reshape(match, [64, 1, 312]), [1, 72, 1])
        score = tf.reduce_sum(tf.multiply(match, K), axis=-1)
        logits = tf.subtract(score, tf.multiply(mask, 999))
        return logits

    @staticmethod
    def loss(labels, logits):
        #print(labels.shape, logits.shape)
        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        return loss

