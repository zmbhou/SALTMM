# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from kaffe.tensorflow import Network
import tensorflow as tf
#from crf import crf_inference


class DeepLabVGGModel(object):
  def __init__(self, inputs, num_classes,is_training):
    self.input = inputs
    #self.num_classes = num_classes
    self.category_num= num_classes
    self.stride = {}
    self.stride["input"] = 1
    self.net={}
    self.net['input']=inputs.get('input')
    self.net["drop_prob"]=0.5
    self.min_prob=0.0001
    self.is_training = is_training

    # For a small batch size, it is better to keep
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model.
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    self.create_network()


  def create_network(self):
    with tf.name_scope("deeplab") as scope:
      block = self.build_block("input", ["conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1"])
      block = self.build_block(block, ["conv2_1", "relu2_1", "conv2_2", "relu2_2", "pool2"])
      block = self.build_block(block, ["conv3_1", "relu3_1", "conv3_2", "relu3_2", "conv3_3", "relu3_3", "pool3"])
      block = self.build_block(block, ["conv4_1", "relu4_1", "conv4_2", "relu4_2", "conv4_3", "relu4_3", "pool4"])
      block = self.build_block(block,
                               ["conv5_1", "relu5_1", "conv5_2", "relu5_2", "conv5_3", "relu5_3", "pool5", "pool5a"])
      fc = self.build_fc(block, ["fc6", "relu6", "drop6", "fc7", "relu7", "drop7", "fc8"])

    with tf.name_scope("sec") as scope:
      softmax = self.build_sp_softmax(fc)
      #crf = self.build_crf(fc, "input")

    self.o=self.net[softmax]

  def build_block(self, last_layer, layer_lists):
    for layer in layer_lists:
      if layer.startswith("conv"):
        if layer[4] != "5":
          with tf.name_scope(layer) as scope:
            print(last_layer)
            self.stride[layer] = self.stride[last_layer]
            weights, bias = self.get_weights_and_bias(layer)
            self.net[layer] = tf.nn.conv2d(self.net[last_layer], weights, strides=[1, 1, 1, 1], padding="SAME",
                                           name="conv")
            self.net[layer] = tf.nn.bias_add(self.net[layer], bias, name="bias")
            last_layer = layer
        if layer[4] == "5":
          with tf.name_scope(layer) as scope:
            self.stride[layer] = self.stride[last_layer]
            weights, bias = self.get_weights_and_bias(layer)
            self.net[layer] = tf.nn.atrous_conv2d(self.net[last_layer], weights, rate=2, padding="SAME", name="conv")
            self.net[layer] = tf.nn.bias_add(self.net[layer], bias, name="bias")
            last_layer = layer
      if layer.startswith("batch_norm"):
        with tf.name_scope(layer) as scope:
          self.stride[layer] = self.stride[last_layer]
          #self.net[layer] = tf.contrib.layers.batch_norm(self.net[last_layer])
          self.net[layer] = tf.contrib.layers.batch_norm(self.net[last_layer], scale=True, activation_fn=None,
                                                         updates_collections=None, is_training=is_trainin, scope=scope)
          last_layer = layer
      if layer.startswith("relu"):
        with tf.name_scope(layer) as scope:
          self.stride[layer] = self.stride[last_layer]
          self.net[layer] = tf.nn.relu(self.net[last_layer], name="relu")
          last_layer = layer
      elif layer.startswith("pool5a"):
        with tf.name_scope(layer) as scope:
          self.stride[layer] = self.stride[last_layer]
          self.net[layer] = tf.nn.avg_pool(self.net[last_layer], ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                                           padding="SAME", name="pool")
          last_layer = layer
      elif layer.startswith("pool"):
        if layer[4] not in ["4", "5"]:
          with tf.name_scope(layer) as scope:
            self.stride[layer] = 2 * self.stride[last_layer]
            self.net[layer] = tf.nn.max_pool(self.net[last_layer], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                             padding="SAME", name="pool")
            last_layer = layer
        if layer[4] in ["4", "5"]:
          with tf.name_scope(layer) as scope:
            self.stride[layer] = self.stride[last_layer]
            self.net[layer] = tf.nn.max_pool(self.net[last_layer], ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                                             padding="SAME", name="pool")
            last_layer = layer
    return last_layer

  def build_fc(self, last_layer, layer_lists):
    for layer in layer_lists:
      if layer.startswith("fc"):
        with tf.name_scope(layer) as scope:
          weights, bias = self.get_weights_and_bias(layer)
          if layer.startswith("fc6"):
            self.net[layer] = tf.nn.atrous_conv2d(self.net[last_layer], weights, rate=12, padding="SAME", name="conv")

          else:
            self.net[layer] = tf.nn.conv2d(self.net[last_layer], weights, strides=[1, 1, 1, 1], padding="SAME",
                                           name="conv")
          self.net[layer] = tf.nn.bias_add(self.net[layer], bias, name="bias")
          last_layer = layer
      if layer.startswith("batch_norm"):
        with tf.name_scope(layer) as scope:
          #self.net[layer] = tf.contrib.layers.batch_norm(self.net[last_layer])
          self.net[layer] = tf.contrib.layers.batch_norm(self.net[last_layer], scale=True, activation_fn=None,
                                                         updates_collections=None, is_training=is_trainin, scope=scope)
          last_layer = layer
      if layer.startswith("relu"):
        with tf.name_scope(layer) as scope:
          self.net[layer] = tf.nn.relu(self.net[last_layer])
          last_layer = layer
      if layer.startswith("drop"):
        with tf.name_scope(layer) as scope:
          self.net[layer] = tf.nn.dropout(self.net[last_layer], self.net["drop_prob"])
          last_layer = layer

    return last_layer

  def build_sp_softmax(self, last_layer):
    layer = "fc8-softmax"
    preds_max = tf.reduce_max(self.net[last_layer], axis=3, keepdims=True)
    preds_exp = tf.exp(self.net[last_layer] - preds_max)
    self.net[layer] = preds_exp / tf.reduce_sum(preds_exp, axis=3, keepdims=True) + self.min_prob
    self.net[layer] = self.net[layer] / tf.reduce_sum(self.net[layer], axis=3, keepdims=True)
    return layer

  def get_weights_and_bias(self, layer):
    if layer.startswith("conv"):
      shape = [3, 3, 0, 0]
      if layer == "conv1_1":
        shape[2] = 3
      else:
        shape[2] = 64 * self.stride[layer]
        if shape[2] > 512: shape[2] = 512
        if layer in ["conv2_1", "conv3_1", "conv4_1"]: shape[2] = int(shape[2] / 2)
      shape[3] = 64 * self.stride[layer]
      if shape[3] > 512: shape[3] = 512
    if layer.startswith("fc"):
      if layer == "fc6":
        shape = [3, 3, 512, 1024]
      if layer == "fc7":
        shape = [1, 1, 1024, 1024]
      if layer == "fc8":
        shape = [1, 1, 1024, self.category_num]

    weights = tf.get_variable(name="%s_weights" % layer, shape=shape)
    bias = tf.get_variable(name="%s_bias" % layer, shape=[shape[-1]])


    return weights, bias

