import tensorflow as tf
import numpy as np
import pickle


class Dataset:
    def __init__(self, filenames, i_ids, batch_size, attention_dict, knowledge_dict, all_data, is_train=True):
        self.filenames = filenames
        self.i_ids = i_ids
        self.batch_size = batch_size
        self.is_train = is_train
        self.attention_dict = attention_dict
        self.knowledge_dict = knowledge_dict
        self.all_data = all_data

    def _parse_function_train(self, filename, i_id):
        feature_map = pickle.load(open(filename[:-4]+'pkl')).reshape((2048, 64)).T.astype(np.float32)
        father = self.all_data[i_id]['layer_coarse']['father']
        attention_vector = self.attention_dict[father].astype(np.float32)
        knowledge = self.knowledge_dict[father].astype(np.float32)
        mask = 1 - self.all_data[i_id]['layer_coarse']['seen_mask'].astype(np.float32)
        label = self.all_data[i_id]['layer_coarse']['label']
        #print(father)
        return feature_map, attention_vector, knowledge, mask, label

    def _parse_function_eval(self, filename, label):
        feature_map = pickle.load(open(filename))['x'].reshape((2048, 64)).T
        name = label['name']
        attention_vector = self.attention_dict[name]
        knowledge = np.attention_dict[name]
        mask = 1 - label['layer_coarse']['unseen_mask']
        label = label['layer_coarse']['label']
        return feature_map, attention_vector, knowledge, mask, label

    @staticmethod
    def _resize_function(f, a, k, m, l):
        f.set_shape([64, 64, 2048])
        a.set_shape([64, 312])
        k.set_shape([64, 72, 312])
        m.set_shape([64, 72])
        l.set_shape([64])
        return {'feature': f, 'att_vec': a, 'knowledge': k, 'mask': m}, l

    def input_fn(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.filenames, self.i_ids))
        if self.is_train:
            dataset = dataset.map(
                lambda filename, i_id: tuple(tf.py_func(
                    self._parse_function_train, [filename, i_id],
                    [tf.float32, tf.float32, tf.float32, tf.float32, tf.int64])))
            dataset = dataset.shuffle(1000).repeat().batch(self.batch_size)
        else:
            dataset = dataset.map(
                lambda filename, label: tuple(tf.py_func(
                    self._parse_function_eval, [filename, label],
                    [tf.float32, tf.float32, tf.float32, tf.float32, label.dtype])))
            dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(self._resize_function)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        return next_element
