import tensorflow as tf
from model import Model
from dataset import Dataset
import pickle
tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '../data/inception_pkl/', """aaa""")
tf.app.flags.DEFINE_integer('max_steps', 100, """lalal""")
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
    train_files, train_labels, test_file, test_labels = load_data()
    train_data = Dataset(train_files, train_labels, 64, attention_dict, knowledge_dict, all_data)
    the_model = Model(lr=0.001)
    the_model.classifier.train(input_fn=train_data.input_fn,
                               hooks=the_model.train_hooks,
                               steps=1000)

#    sess = tf.Session()
#    h = tf.constant(0,dtype=tf.int64)
#    m, b = train_data.input_fn()
#    x = sess.run([m, b+h])
#    print(x)
#    predict_result = the_model.classifier.predict(input_fn=lambda: input_fn(test_files, test_labels, istrain=False))
#    predict_result = list(predict_result)
#    count = 0
#    for i in range(len(predict_result)):
#        if predict_result[i]['class_ids'][0] == test_labels[i]:
#            count += 1
#    print(count, len(test_labels))


if __name__ == '__main__':
    tf.app.run()
