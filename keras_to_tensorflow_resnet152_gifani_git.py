# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 15:12:57 2019

@author: Gifani
"""


import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from pathlib import Path
from absl import app
from absl import flags
from absl import logging
import keras
from keras import backend as K
from keras.models import model_from_json
from tensorflow.python.tools import freeze_graph
from keras import optimizers
from classification_models.resnet import ResNet152, preprocess_input




model_weights_path = 'root_for.hdf5'

out_model_pb="root_for.pb"


num_classes=N


K.set_learning_phase(0)
FLAGS = flags.FLAGS

flags.DEFINE_string('input_model', '', 'Path to the input model.')
flags.DEFINE_string('input_model_json', None, 'Path to the input model '
                                              'architecture in json format.')
flags.DEFINE_string('output_model',out_model_pb , 'Path where the converted model will '
                                          'be stored.')
flags.DEFINE_boolean('save_graph_def', True,
                     'Whether to save the graphdef.pbtxt file which contains '
                     'the graph definition in ASCII format.')
flags.DEFINE_string('output_nodes_prefix', None,
                    'If set, the output nodes will be renamed to '
                    '`output_nodes_prefix`+i, where `i` will numerate the '
                    'number of of output nodes of the network.')
flags.DEFINE_boolean('quantize', False,
                     'If set, the resultant TensorFlow graph weights will be '
                     'converted from float into eight-bit equivalents. See '
                     'documentation here: '
                     'https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms')
flags.DEFINE_boolean('channels_first', False,
                     'Whether channels are the first dimension of a tensor. '
                     'The default is TensorFlow behaviour where channels are '
                     'the last dimension.')
flags.DEFINE_boolean('output_meta_ckpt', True,
                     'If set to True, exports the model as .meta, .index, and '
                     '.data files, with a checkpoint file. These can be later '
                     'loaded in TensorFlow to continue training.')


def load_model_car():
    
    base_model = ResNet152( input_shape=(224,224,3), weights='imagenet', include_top=False)
    base_model.layers[0].name ='data_1'

    x = keras.layers.AveragePooling2D((7,7))(base_model.output)

    x_newfc = keras.layers.Flatten()(x)

    x_newfc = keras.layers.Dense(num_classes, activation='softmax', name='fc_new')(x_newfc)

    model_car = keras.models.Model(input=base_model.input, output=x_newfc)
    
    sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    
    model_car.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    model_car.load_weights(model_weights_path, by_name=True)
    
    return model_car



print('loading model...')

def main(args):
    
    # If output_model path is relative and in cwd, make it absolute from root
    output_model = FLAGS.output_model
    if str(Path(output_model).parent) == '.':
        output_model = str((Path.cwd() / output_model))

    output_fld = Path(output_model).parent
    #input_fld=Path()
    output_model_name = Path(output_model).name
    output_model_stem = Path(output_model).stem
    output_model_pbtxt_name = output_model_stem + '.pbtxt'

    # Create output directory if it does not exist
    Path(output_model).parent.mkdir(parents=True, exist_ok=True)

    if FLAGS.channels_first:
        K.set_image_data_format('channels_first')
    else:
        K.set_image_data_format('channels_last')

    model = load_model_car()
    model.summary()
    orig_input_node_names = [node.op.name for node in model.inputs]
    
    
    print("***********:" + orig_input_node_names[0])
    # TODO(amirabdi): Support networks with multiple inputs
    orig_output_node_names = [node.op.name for node in model.outputs]
    if FLAGS.output_nodes_prefix:
        num_output = len(orig_output_node_names)
        pred = [None] * num_output
        converted_output_node_names = [None] * num_output

        # Create dummy tf nodes to rename output
        for i in range(num_output):
            converted_output_node_names[i] = '{}{}'.format(
                FLAGS.output_nodes_prefix, i)
            pred[i] = tf.identity(model.outputs[i],
                                  name=converted_output_node_names[i])
    else:
        converted_output_node_names = orig_output_node_names
    logging.info('Converted output node names are: %s',
                 str(converted_output_node_names))
        
    sess = K.get_session()
    if FLAGS.output_meta_ckpt:
        saver = tf.train.Saver()
        saver.save(sess, str(output_fld / output_model_stem))
    
    if FLAGS.save_graph_def:
        tf.train.write_graph(sess.graph.as_graph_def(), str(output_fld),
                             output_model_pbtxt_name, as_text=True)
        logging.info('Saved the graph definition in ascii format at %s',
                     str(Path(output_fld) / output_model_pbtxt_name))
    


    if FLAGS.quantize:
        from tensorflow.tools.graph_transforms import TransformGraph
        transforms = ["quantize_weights", "quantize_nodes"]
        transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [],
                                               converted_output_node_names,
                                               transforms)
        constant_graph = graph_util.convert_variables_to_constants(
            sess,
            transformed_graph_def,
            converted_output_node_names)
    else:
        constant_graph = graph_util.convert_variables_to_constants(
            sess,
            
            sess.graph.as_graph_def(),
            converted_output_node_names)

    graph_io.write_graph(constant_graph, str(output_fld), output_model_name,
                         as_text=False)
    logging.info('Saved the freezed graph at %s',
                 str(Path(output_fld) / output_model_name))


if __name__ == "__main__":
    app.run(main)



























