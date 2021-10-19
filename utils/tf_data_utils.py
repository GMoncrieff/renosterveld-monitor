import tensorflow as tf
from tensorflow.keras.losses import Loss, Reduction
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math

from .globals import *

#window observations into set length
def shift_window(x,shift,band_len):
  
  x = [tf.squeeze(val) for val in list(x.values())]
  x = tf.transpose(tf.convert_to_tensor(x))

  tf_window = tf.data.Dataset.from_tensor_slices(x)\
  .window(WINDOW,shift=int(shift),stride=1, drop_remainder=True)

  tf_out= tf_window.flat_map(lambda window: window.batch(WINDOW))\
  .map(lambda tens: tf.reshape(tf.transpose(tens),[band_len,WINDOW]))

  return tf_out

#scale to 0-1
def standardize(x,Min,Max):
  for k in x:
    x[k] = (x[k]-Min[k])/(Max[k]-Min[k])
  return x

def roller(x):
  z = tf.roll(x['label'],2,0)
  x['lablag'] = tf.math.add(z,x['label'])
  return x

#reshape data from dict into tuple of features and label
def reshape_ts(x,names,extra=[]):
  lab_tup = sorted(list(names))+extra
  val_tup = list(tf.unstack(x,axis=0))
  res = dict(zip(lab_tup,val_tup))
  return res

#set labbel as most recent label
def labeller(x,lab):
  x[lab] = [tf.cast(x[lab][-1], tf.int32)]
  return x

#remove unwanted labels
def filter_fn(x,allowed_labels=LABELS_FILTER,label=label):
    allowed_labels=tf.constant(allowed_labels)
    lab = x[label]
    isallowed = tf.equal(allowed_labels, tf.cast(lab, tf.float32))
    reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
    return tf.greater(reduced, tf.constant(0.))

#remove label from features
def popkey(x,keys): 
    for e in keys: 
        x.pop(e)
    return x

#remove label from features
def poplabint(x,labels): 
    lab = x[labels]
    lab = tf.cast(lab,tf.int32)
    x.pop(labels, None)   
    return (x,lab)

# Keras requires inputs as a tuple.  Note that the inputs must be in the
# right shape.  Also note that to use the categorical_crossentropy loss,
# the label needs to be turned into a one-hot vector.

def toTuple(x, lab):
    return tf.transpose(list(x.values())), tf.squeeze(tf.one_hot(indices=tf.dtypes.cast(lab, tf.int32), depth=n_classes))

# transpose function.
def tp_image(tf_Data):
  for k, v in tf_Data.items():
    tf_Data[k] = tf.transpose(v)
  return tf_Data

def parse_tfrecord(example_proto,featuresDict):
  """The parsing function.

  Read a serialized example into the structure defined by featuresDict.

  Args:
    example_proto: a serialized Example.
  
  Returns: 
    A tuple of the predictors dictionary and the label, cast to an `int32`.
  """
  parsed_features = tf.io.parse_single_example(example_proto, featuresDict)
 # labels = parsed_features.pop(label)
 # return parsed_features, tf.cast(labels, tf.int32)
  return parsed_features

def parse_image(example_proto,imDict):
  return tf.io.parse_single_example(example_proto, imDict)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    

class CategoricalFocalLoss(Loss):
    """ Categorical version of focal loss.
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        Keras implementation: https://github.com/umbertogriffo/focal-loss-keras
    """

    def __init__(self, gamma=2., alpha=.25, from_logits=True, class_weights=None, reduction=Reduction.AUTO,
                 name='FocalLoss'):
        """Categorical version of focal loss.
        :param gamma: gamma value, defaults to 2.
        :type gamma: float
        :param alpha: alpha value, defaults to .25
        :type alpha: float
        :param from_logits: Whether predictions are logits or softmax, defaults to True
        :type from_logits: bool
        :param class_weights: Array of class weights to be applied to loss. Needs to be of `n_classes` length
        :type class_weights: np.array
        :param reduction: reduction to be used, defaults to Reduction.AUTO
        :type reduction: tf.keras.losses.Reduction, optional
        :param name: name of the loss, defaults to 'FocalLoss'
        :type name: str
        """
        super().__init__(reduction=reduction, name=name)

        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits
        self.class_weights = class_weights

    def call(self, y_true, y_pred):

        # Perform softmax
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.)

        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate Focal Loss
        loss = self.alpha * tf.math.pow(1 - y_pred, self.gamma) * cross_entropy

        # Multiply focal loss with class-wise weights
        if self.class_weights is not None:
            loss = tf.multiply(cross_entropy, self.class_weights)

        # Sum over classes
        loss = tf.reduce_sum(loss, axis=-1)

        return loss