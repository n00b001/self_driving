import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import create_feature_spec_for_parsing
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.contrib.predictor import from_saved_model
from tensorflow.core.protobuf.meta_graph_pb2 import SignatureDef
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def
from tensorflow.python.saved_model.utils_impl import build_tensor_info

from x2_train_net import x


def save_tf_learn_model(estimator, model_name, export_dir, feature_columns):
    feature_spec = create_feature_spec_for_parsing(feature_columns)
    serving_input_fn = input_fn_utils.build_parsing_serving_input_fn(feature_spec)
    export_dir = os.path.join(export_dir, model_name)
    estimator.export_savedmodel(export_dir, serving_input_fn)
    print("Done exporting tf.learn model to " + export_dir + "!")


SESS_DICT = {}


def get_session(model_id):
    global SESS_DICT
    config = tf.ConfigProto(allow_soft_placement=True)
    SESS_DICT[model_id] = tf.Session(config=config)
    return SESS_DICT[model_id]


def load_tf_model(model_path):
    sess = get_session(model_path)
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
    return sess


# feature_spec = {'x': tf.FixedLenFeature(shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH), dtype=tf.float32)}


def serving_input_receiver_fn(feature_columns):
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()
    #
    # screen = tf.placeholder(dtype=tf.float32, shape=[IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH], name='x')
    # receiver_tensors = {'x': screen}
    # features = tf.tile(screen, multiples=[1, 2, 3])
    # return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    # """An input receiver that expects a serialized tf.Example."""
    # serialized_tf_example = tf.placeholder(dtype=tf.float32,
    #                                        # shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH],
    #                                        name='x')
    # receiver_tensors = {'examples': serialized_tf_example}
    # features = tf.parse_example(serialized_tf_example, feature_spec)
    # return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


# def save_model(export_path):
#     builder = saved_model.builder.SavedModelBuilder(export_path)
#
#     signature = predict_signature_def(inputs={'myInput': x},
#                                       outputs={'myOutput': y})
#     # using custom tag instead of: tags=[tag_constants.SERVING]
#     builder.add_meta_graph_and_variables(sess=sess,
#                                          tags=["myTag"],
#                                          signature_def_map={'predict': signature})
#     builder.save()


def get_latest(folder):
    all_subdirs = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    latest = max(all_subdirs, key=os.path.getmtime)
    print("Found latest: {}".format(latest))
    return latest


def run_saved_model(_):
    model_dir = get_latest("models/WRKHZOWXNC")
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # model_dir = "models/KEWXRVEADT/1556810329"
    #
    # tensor_info_x = build_tensor_info(x)
    # # tensor_info_y = build_tensor_info(y)
    #
    # sig = build_signature_def(
    #     inputs={'x': tensor_info_x},
    #     # outputs={'output': tensor_info_y},
    #     # method_name=signature_constants.PREDICT_METHOD_NAME
    # )
    print("Loading model...")
    predict_fn = from_saved_model(model_dir, signature_def_key='predict')
    # predict_fn = from_saved_model(model_dir, signature_def=sig)#, signature_def_key='predict')

    graph = tf.get_default_graph()
    print(graph.get_operations())

    zeros = np.zeros((96, 96, 3,), dtype=np.float32)
    # inputs = pd.DataFrame({
    #     'SepalLength': [5.1, 5.9, 6.9],
    #     'SepalWidth': [3.3, 3.0, 3.1],
    #     'PetalLength': [1.7, 4.2, 5.4],
    #     'PetalWidth': [0.5, 1.5, 2.1],
    # })
    #
    # zeros_feat = tf.train.Feature(float_list=tf.train.FloatList(value=zeros))
    # example = tf.train.Example(
    #     features=tf.train.Features(
    #         feature=zeros_feat
    #     ))
    # # Convert input data into serialized Example strings.
    # examples = []
    # for index, row in inputs.iterrows():
    #     feature = {}
    #     for col, value in row.iteritems():
    #         feature[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    #     example = tf.train.Example(
    #         features=tf.train.Features(
    #             feature=feature
    #         )
    #     )
    #     examples.append(example.SerializeToString())

    print("Predicting...")
    pred = predict_fn({
        'x': [zeros]
    })
    print(pred)
    # # Example message for inference
    # message = "Was ist denn los"
    # saved_model_predictor = from_saved_model(export_dir=model_dir)
    # content_tf_list = tf.train.BytesList(value=[message.encode('utf-8')])
    # sentence = tf.train.Feature(bytes_list=content_tf_list)
    # sentence_dict = {'sentence': sentence}
    # features = tf.train.Features(feature=sentence_dict)
    #
    # example = tf.train.Example(features=features)
    #
    # serialized_example = example.SerializeToString()
    # output_dict = saved_model_predictor({'inputs': [serialized_example]})

    # with tf.Session(graph=tf.Graph()) as sess:
    #     tf.saved_model.loader.load(sess, ["serve"], model_dir)
    #
    #     graph = tf.get_default_graph()
    #     print(graph.get_operations())
    #     zeros = np.zeros((1, 96, 96, 3), dtype=np.float32)
    #     # zeros = tf.parse_tensor(zeros, np.float32)
    #     input = {"x:0": zeros}
    #     # input = tf.parse_tensor(input)
    #     resp = sess.run('dnn/head/Tile:0', feed_dict=input)
    #     print(resp)


    # thing = {"x": np.expand_dims(np.zeros((96, 96, 3), dtype=np.float32), axis=0)}
    # with open('thing.pkl', 'wb') as f:
    #     pickle.dump(thing, f, pickle.HIGHEST_PROTOCOL)
    #
if __name__ == '__main__':
    # run_saved_model()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(run_saved_model)
