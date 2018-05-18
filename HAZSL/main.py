import tensorflow as tf
from model import Model
from dataset import Dataset
import pickle
import utils
tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '../data/inception_pkl/', """aaa""")
tf.app.flags.DEFINE_integer('max_steps', 100, """lalal""")
tf.app.flags.DEFINE_integer('epoch', 1000, """lalal""")
tf.app.flags.DEFINE_string('model_dir', './model/', """aaa""")
tf.app.flags.DEFINE_string('CUB_data', '../CUB_hierarchy/', """sess""")

def load_data():
    filenames = []
    i_ids = []
    with open(FLAGS.CUB_data + 'trainimages.txt', 'rb') as train_images:
        for line in train_images.readlines():
            i_id, i_name = line.split(' ')
            filenames.append(FLAGS.train_dir + i_name)
            i_ids.append(int(i_id))

    test_files = []
    test_i_ids = []
    with open(FLAGS.CUB_data + 'testimages.txt', 'rb') as test_images:
        for line in test_images.readlines():
            i_id, i_name = line.split(' ')
            test_files.append(FLAGS.train_dir + i_name)
            test_i_ids.append(int(i_id))
    return filenames, i_ids, test_files, test_i_ids


def main(argv=None):
    all_data = pickle.load(open(FLAGS.CUB_data + 'cub_image_dict.pkl', 'r'))
    attention_dict = pickle.load(open(FLAGS.CUB_data + 'cub_attention_dict.pkl', 'r'))
    knowledge_dict = pickle.load(open(FLAGS.CUB_data + 'cub_knowledge_dict.pkl', 'r'))
    train_files, train_labels, test_files, test_labels = load_data()
    

    test_f = []
    test_l = []
    test_fine_f = []
    test_fine_l = []
    for i in xrange(len(test_labels)):
        if 'layer_fine' in all_data[test_labels[i]].keys():
        #    continue
            test_f.append(test_files[i])
            test_l.append(test_labels[i])
            test_fine_f.append(test_files[i] + 'f')
            test_fine_l.append(test_labels[i])
        else:
            test_f.append(test_files[i])
            test_l.append(test_labels[i])

    train_data = Dataset(train_files, train_labels, 64, attention_dict, knowledge_dict, all_data)
    test_data = Dataset(test_f, test_l, 64, attention_dict, knowledge_dict, all_data, is_train=False)
    test_data_fine = Dataset(test_fine_f, test_fine_l, 64, attention_dict, knowledge_dict, all_data, is_train=False)
    the_model = Model(lr=0.002)
    fine_predict = the_model.classifier.predict(input_fn=test_data_fine.input_fn)
    coarse_predict = the_model.classifier.predict(input_fn=test_data.input_fn)
    accuracy = utils.calc_accuracy(test_l, fine_predict, coarse_predict, all_data)
    print(accuracy)
    #the_model.classifier.evaluate(input_fn=test_data.input_fn)
    '''
    for i in xrange(FLAGS.epoch):
        the_model.classifier.evaluate(input_fn=test_data_fine.input_fn)
        fine_predict = the_model.classifier.predict(input_fn=test_data_fine.input_fn)
        coarse_predict = the_model.classifier.predict(input_fn=test_data.input_fn)
        the_model.classifier.train(input_fn=train_data.input_fn,
                                hooks=the_model.train_hooks,
                                steps=200)
        the_model.classifier.evaluate(input_fn=test_data.input_fn)
    '''
    '''
    sess = tf.Session()
    h = tf.constant(0,dtype=tf.int64)
    #w = tf.get_variable('w', shape=(2,2))
    #sess.run(tf.global_variables_initializer())
    m, b = train_data.input_fn()
    #q = tf.trainable_variables()
    #f = tf.contrib.layers.l2_regularizer(1.0)
    #e = tf.contrib.layers.apply_regularization(f,weights_list=q)
    #p = tf.reduce_sum(tf.reduce_sum(tf.multiply(w, w)))
    x = sess.run([m,b+h])
    print(x)
    '''
#    predict_result = the_model.classifier.predict(input_fn=lambda: input_fn(test_files, test_labels, istrain=False))
#    predict_result = list(predict_result)
#    count = 0
#    for i in range(len(predict_result)):
#        if predict_result[i]['class_ids'][0] == test_labels[i]:
#            count += 1
#    print(count, len(test_labels))


if __name__ == '__main__':
    tf.app.run()
