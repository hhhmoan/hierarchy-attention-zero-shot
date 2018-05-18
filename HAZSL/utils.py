import tensorflow as tf


def look_embedding(look_up_table, idx):
    """
    :param look_up_table:
    :param idx:
    :return: look_up_table[idx]
    """
    return tf.nn.embedding_lookup(look_up_table, idx)


def batch_normalization(inputs, training=True, name='bn'):
    """
    :param inputs:
    :param training:
    :param name:
    :return: batch_normalization
    """
    outputs = tf.layers.batch_normalization(inputs=inputs,
                                            axis=-1,
                                            momentum=0.99,
                                            epsilon=0.001,
                                            center=True,
                                            scale=True,
                                            training=training,
                                            name=name)
    return outputs


def fully_connect(inputs, units, activation=tf.nn.relu, use_bias=True, trainable=True, name='fc', reuse=None):
    """
    :param inputs:
    :param units:
    :param activation:
    :param use_bias:
    :param trainable:
    :param name:
    :return: output = fc(inputs)
    """
    outputs = tf.layers.dense(inputs=inputs,
                              units=units,
                              activation=activation,
                              use_bias=use_bias,
                              trainable=trainable,
                              name=name,
                              reuse=reuse)
    return outputs


def dropout(inputs, rate=0.5, training=True, name='dropout'):
    """
    :param inputs:
    :param rate:
    :param training:
    :param name:
    :return: dropout(inputs)
    """
    outputs = tf.layers.dropout(inputs=inputs,
                                rate=rate,
                                training=training,
                                name=name)
    return outputs


def conv2d(inputs, filters, kernel_size, strides=(1,1), padding='valid', data_format='channels_last'
           , activation=None, use_bias=True, trainable=True, name='conv2d', reuse=None):
    """
    :param inputs:
    :param filters:
    :param kernel_size:
    :param strides:
    :param padding:
    :param data_format:
    :param activation:
    :param use_bias:
    :param trainable:
    :param name:
    :param reuse:
    :return: output = conv2d(x)
    """
    outputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                               strides=strides, padding=padding, data_format=data_format,
                               activation=activation, use_bias=use_bias, trainable=trainable,
                               name=name, reuse=reuse)
    return outputs


def deconv2d(inputs, filters, kernel_size, strides=(1,1), padding='valid', data_format='channels_last'
             , activation=None, use_bias=True, trainable=True, name='deconv2d', reuse=None):
    """
    :param inputs:
    :param filters:
    :param kernel_size:
    :param strides:
    :param padding:
    :param data_format:
    :param activation:
    :param use_bias:
    :param trainable:
    :param name:
    :param reuse:
    :return: output = deconv2d(x)
    """
    outputs = tf.layers.conv2d_transpose(inputs=inputs, filters=filters, kernel_size=kernel_size,
                                         strides=strides, padding=padding, data_format=data_format,
                                         activation=activation, use_bias=use_bias, trainable=trainable,
                                         name=name, reuse=reuse)
    return outputs


def attention_vector(x, vec_length):
    """
    :param x:
    :param vec_length:
    :return: f(x) to vec_length
    """
    x = tf.layers.dense(x, vec_length, activation=tf.nn.relu)
    return x


def attention_layer(feature_map, x):
    """
    :param feature_map:
    :param x:
    :return: use x to attention feature_map
    {batch_size, feature_map_size, channels}  {batch_size, channels}
    """
    s = tf.expand_dims(x, 1)
    s = tf.tile(s, [1, feature_map.shape[1], 1])
    s = tf.multiply(tf.nn.l2_normalize(feature_map,2), s)
    s = tf.reduce_sum(s, axis=-1)
    attention = tf.nn.softmax(s, dim=-1)
    attention = tf.expand_dims(attention, -1)
    attention = tf.tile(attention, [1, 1, feature_map.shape[-1]])
    feature = tf.reduce_sum(tf.multiply(feature_map, attention), axis=1)
    return feature


def transform2knowledge(feature, knowledge_length):
    knowledge = tf.layers.dense(feature, knowledge_length, activation=tf.nn.relu)
    return knowledge

def calc_accuracy(test_l, fine_predict, coarse_predict, all_data):
    fine_predict = list(fine_predict)
    coarse_predict = list(coarse_predict)
    #print(fine_predict)
    print(len(fine_predict))
    assert len(fine_predict) == len(coarse_predict)
    assert len(fine_predict) == len(test_l)
    count = 0.0
    fine_acc = 0.0
    coarse_acc = 0.0
    for i in range(len(test_l)):
        fine_label = all_data[test_l[i]]['layer_fine']['label']
        coarse_label = all_data[test_l[i]]['layer_coarse']['label']
        if fine_predict[i]['class_ids'][0] == fine_label and coarse_predict[i]['class_ids'][0] == coarse_label:
            count += 1.0
        if fine_predict[i]['class_ids'][0] == fine_label:
            fine_acc += 1.0
        if coarse_predict[i]['class_ids'][0] == coarse_label:
            coarse_acc += 1.0
    return count / len(test_l), fine_acc / len(test_l), coarse_acc / len(test_l)
