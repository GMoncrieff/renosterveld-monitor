#window observations into set length
def flat_window(x:
  
  x = [tf.squeeze(val) for val in list(x.values())]
  x = tf.transpose(tf.convert_to_tensor(x))

  tf_window = tf.data.Dataset.from_tensor_slices(x)\
  .window(WINDOW,shift=int(WINDOW/3),stride=1, drop_remainder=True)

  tf_out= tf_window.flat_map(lambda window: window.batch(WINDOW))\
  .map(lambda tens: tf.reshape(tf.transpose(tens),[len(allbands)+1,WINDOW]))

  return tf_out

#scale to 0-1
def standardize(x):
  for k in x:
    x[k] = (x[k]-Min[k])/(Max[k]-Min[k])
  return x

#reshape data from dict into tuple of features and label
def reshape_ts(x):
  lab_tup = sorted(list(featureNames))
  val_tup = list(tf.unstack(x,axis=0))
  
  res = dict(zip(lab_tup,val_tup))

  return res

#reshape data from dict into tuple of features and label
def reshape_test(x):
  lab_tup = sorted(list(testNames))
  val_tup = list(tf.unstack(x,axis=0))
  
  res = dict(zip(lab_tup,val_tup))

  return res

#set labbel as most recent label
def labeller(x):
  x[label] = [tf.cast(x[label][-1], tf.int32)]
  return x

#set labbel as most recent label
def labeller_roll(x):
  x['roll'] = [tf.cast(x['roll'][-1], tf.int32)]
  return x

#remove unwanted labels
def filter_fn(x,allowed_labels=[0.0,1.0]):
    allowed_labels=tf.constant(allowed_labels)
    label = x['label']
    isallowed = tf.equal(allowed_labels, tf.cast(label, tf.float32))
    reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
    return tf.greater(reduced, tf.constant(0.))

#needs these bc I cant figure out how to pass variable to filter func
def filter_fn0(x,allowed_labels=[0.0]):
    allowed_labels=tf.constant(allowed_labels)
    label = x['label']
    isallowed = tf.equal(allowed_labels, tf.cast(label, tf.float32))
    reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
    return tf.greater(reduced, tf.constant(0.))
def filter_fn1(x,allowed_labels=[1.0]):
    allowed_labels=tf.constant(allowed_labels)
    label = x['label']
    isallowed = tf.equal(allowed_labels, tf.cast(label, tf.float32))
    reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
    return tf.greater(reduced, tf.constant(0.))

#class weighting
def class_weighting(inpath):
    
    dataset0 = tf.data.TFRecordDataset(inpath, compression_type='GZIP')\
    .map(parse_tfrecord, num_parallel_calls=npar)\
    .flat_map(flat_window)\
    .map(reshape_ts)\
    .map(labeller)\
    .filter(filter_fn0)

    dataset1 = tf.data.TFRecordDataset(inpath, compression_type='GZIP')\
    .map(parse_tfrecord, num_parallel_calls=npar)\
    .flat_map(flat_window)\
    .map(reshape_ts)\
    .map(labeller)\
    .filter(filter_fn1)
    
    #check dataset len
    num_elements0 = 0
    for element in dataset0:
        num_elements0 += 1
    train_len0 = num_elements0

    #check dataset len
    num_elements1 = 0
    for element in dataset1:
        num_elements1 += 1
    train_len1 = num_elements1
    
    total = train_len0+train_len1
    print(str(total),str(train_len0),str(train_len1))
    
    weight_for_0 = (1 / train_len0)*(total)/2.0 
    weight_for_1 = (1 / train_len1)*(total)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    return class_weight

#remove label from features
def popkey(x): 
    x.pop('ndvi_diff', None)
    x.pop('roll', None) 
    x.pop('count', None)   
    return x


#return only label
def poproll(x):
    x = x['roll']  
    return (x)

def poplabels(x):
    x = x['label']  
    return (x)

#remove label from features
def poplab(x): 
    lab = x[label]
    x.pop(label, None)   
    return (x,lab)

def poplabint(x): 
    lab = x[label]
    lab = tf.cast(lab,tf.int32)
    x.pop(label, None)   
    return (x,lab)

def poprollint(x): 
    lab = x['roll']
    lab = tf.cast(lab,tf.int32)
    x.pop('roll', None) 
    x.pop('label',None)  
    return (x,lab)

#remove label from features confusion mat
def poplabcm(x,lab):
  #lab = tf.math.argmax(lab)
  return lab

# Keras requires inputs as a tuple.  Note that the inputs must be in the
# right shape.  Also note that to use the categorical_crossentropy loss,
# the label needs to be turned into a one-hot vector.

def toTupleOld(dictionary, label='label'):
  return tf.transpose(list(dictionary.values())), tf.squeeze(tf.one_hot(indices=label, depth=nClasses))

def toTuple(dictionary, label='label'):
      return tf.transpose(list(dictionary.values())), tf.squeeze(tf.one_hot(indices=tf.dtypes.cast(label, tf.int32), depth=nClasses))

def toTupleRoll(dictionary, label='roll'):
      return tf.transpose(list(dictionary.values())), tf.squeeze(tf.one_hot(indices=tf.dtypes.cast(label, tf.int32), depth=nClasses))


def toTuple_nothotOld(dictionary, label='label'):
  return tf.transpose(list(dictionary.values())), label

def toTuple_nothot(dictionary, label='label'):
      return tf.transpose(list(dictionary.values())), tf.dtypes.cast(label, tf.int32)

def arr2dict(arr, label):
    res = {"x" : arr} 
    return res, label

def flatarr(arr, label):
    #dims = tf.shape(arr).numpy()
    #alen = dims[0]*dims[1]
    return tf.reshape(arr,[648]), label

# transpose function.
def tp_image(tf_Data):
  for k, v in tf_Data.items():
    tf_Data[k] = tf.transpose(v)
  return tf_Data

def parse_tfrecord(example_proto):
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