import ee
service_account = 'renosterveld-ee@ee-vegetation-gee4geo.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, '../ee-vegetation-gee4geo-6309a79ef209.json')
ee.Initialize(credentials)
import tensorflow as tf
import json
from eoflow.models import TransformerEncoder, TempCNNModel
from utils.tf_data_utils import *
from utils.globals import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def model_cnn(params):


    #load pars for scaling data
    with open('max.json') as f:
      Max = json.load(f)

    with open('min.json') as f:
      Min = json.load(f)

    #load data
    trainFilePath = 'data/Training_reno_cnn.tfrecord.gz'
    validFilePath = 'data/Valid_reno_cnn.tfrecord.gz'

    # List of fixed-length features, all of which are float32.
    columns = [
      tf.io.FixedLenFeature(shape=[tsLength], dtype=tf.float32) 
        for k in featureNames
    ]
    # Dictionary with names as keys, features as values.
    featuresDict = dict(zip(featureNames, columns))

    dataset = tf.data.TFRecordDataset(trainFilePath, compression_type='GZIP')\
    .map(lambda x: parse_tfrecord(x,featuresDict))\
    .flat_map(lambda x: shift_window(x, WINDOW/3, len(allbands)+1))\
    .map(lambda x: reshape_ts(x, featureNames))\
    .map(lambda x: labeller(x, label))\
    .filter(filter_fn)\
    .map(lambda x: standardize(x, Min,Max))\
    .map(lambda x: poplabint(x, label))\
    .map(lambda x, lab: toTuple(x, lab))\
    .shuffle(buffer_size=buff, seed=42)\
    .repeat()\
    .batch(batch_size)

    valDataset = tf.data.TFRecordDataset(validFilePath, compression_type='GZIP')\
    .map(lambda x: parse_tfrecord(x,featuresDict))\
    .flat_map(lambda x: shift_window(x, WINDOW/3, len(allbands)+1))\
    .map(lambda x: reshape_ts(x, featureNames))\
    .map(lambda x: labeller(x, label))\
    .filter(filter_fn)\
    .map(lambda x: standardize(x, Min,Max))\
    .map(lambda x: poplabint(x, label))\
    .map(lambda x, lab: toTuple(x, lab))\
    .batch(batch_size)

    #model pars
    model_cfg = {
        "learning_rate": lr,
        "n_classes": n_classes,
        "keep_prob": params[0],
        "metrics": ["accuracy","precision","recall"],
        "loss": "focal_loss",
        "kernel_regularizer": params[1],
        "nb_conv_filters": params[2],
        "nb_conv_stacks": params[3]
    }

    # rubn model
    model = TempCNNModel(model_cfg)
    model.build(input_shape)
    model.prepare()
    model.train_and_evaluate(
        train_dataset=dataset,
        val_dataset=valDataset,
        num_epochs=epochs,
        iterations_per_epoch=steps,
        model_directory=model_directory,
        save_steps='epoch',
        summary_steps='epoch',
        verbose=0
    )
    #get results
    result = model.evaluate(valDataset,verbose=0,return_dict=True)
    
    return result

def model_trans(params):


    #load pars for scaling data
    with open('max.json') as f:
      Max = json.load(f)

    with open('min.json') as f:
      Min = json.load(f)

    #load data
    trainFilePath = 'data/Training_reno_cnn.tfrecord.gz'
    validFilePath = 'data/Valid_reno_cnn.tfrecord.gz'

    # List of fixed-length features, all of which are float32.
    columns = [
      tf.io.FixedLenFeature(shape=[tsLength], dtype=tf.float32) 
        for k in featureNames
    ]
    # Dictionary with names as keys, features as values.
    featuresDict = dict(zip(featureNames, columns))

    dataset = tf.data.TFRecordDataset(trainFilePath, compression_type='GZIP')\
    .map(lambda x: parse_tfrecord(x,featuresDict))\
    .flat_map(lambda x: shift_window(x, WINDOW/3, len(allbands)+1))\
    .map(lambda x: reshape_ts(x, featureNames))\
    .map(lambda x: labeller(x, label))\
    .filter(filter_fn)\
    .map(lambda x: standardize(x, Min,Max))\
    .map(lambda x: poplabint(x, label))\
    .map(lambda x, lab: toTuple(x, lab))\
    .shuffle(buffer_size=buff, seed=42)\
    .repeat()\
    .batch(batch_size)

    valDataset = tf.data.TFRecordDataset(validFilePath, compression_type='GZIP')\
    .map(lambda x: parse_tfrecord(x,featuresDict))\
    .flat_map(lambda x: shift_window(x, WINDOW/3, len(allbands)+1))\
    .map(lambda x: reshape_ts(x, featureNames))\
    .map(lambda x: labeller(x, label))\
    .filter(filter_fn)\
    .map(lambda x: standardize(x, Min,Max))\
    .map(lambda x: poplabint(x, label))\
    .map(lambda x, lab: toTuple(x, lab))\
    .batch(batch_size)

    #model pars
    model_cfg = {
        "learning_rate": lr/2,
        "n_classes": n_classes,
        "keep_prob": params[0],
        "metrics": ["accuracy","precision","recall"],
        "loss": "focal_loss",
        "num_heads": params[1],
        "num_layers": params[2],
        "d_model": params[3],
        "layer_norm": params[4],
        "num_dff":params[5]
    }

    # rubn model
    model = TransformerEncoder(model_cfg)
    model.build(input_shape)
    model.prepare()
    model.train_and_evaluate(
        train_dataset=dataset,
        val_dataset=valDataset,
        num_epochs=epochs,
        iterations_per_epoch=steps,
        model_directory=model_directory,
        save_steps='epoch',
        summary_steps='epoch',
        verbose=0
    )
    #get results
    result = model.evaluate(valDataset,verbose=0,return_dict=True)
    
    return result

def model_rf_data():

    #load pars for scaling data
    with open('max.json') as f:
      Max = json.load(f)

    with open('min.json') as f:
      Min = json.load(f)

    #load data
    trainFilePath = 'data/Training_reno_cnn.tfrecord.gz'
    validFilePath = 'data/Valid_reno_cnn.tfrecord.gz'

    # List of fixed-length features, all of which are float32.
    columns = [
      tf.io.FixedLenFeature(shape=[tsLength], dtype=tf.float32) 
        for k in featureNames
    ]
    # Dictionary with names as keys, features as values.
    featuresDict = dict(zip(featureNames, columns))

    dataset = tf.data.TFRecordDataset(trainFilePath, compression_type='GZIP')\
    .map(lambda x: parse_tfrecord(x,featuresDict))\
    .flat_map(lambda x: shift_window(x, WINDOW/3, len(allbands)+1))\
    .map(lambda x: reshape_ts(x, featureNames))\
    .map(lambda x: labeller(x, label))\
    .filter(filter_fn)\
    .map(lambda x: standardize(x, Min,Max))\
    .map(lambda x: poplabint(x, label))\
    .map(lambda x, lab: toTuple(x, lab))\
    .batch(1)

    valDataset = tf.data.TFRecordDataset(validFilePath, compression_type='GZIP')\
    .map(lambda x: parse_tfrecord(x,featuresDict))\
    .flat_map(lambda x: shift_window(x, WINDOW/3, len(allbands)+1))\
    .map(lambda x: reshape_ts(x, featureNames))\
    .map(lambda x: labeller(x, label))\
    .filter(filter_fn)\
    .map(lambda x: standardize(x, Min,Max))\
    .map(lambda x: poplabint(x, label))\
    .map(lambda x, lab: toTuple(x, lab))\
    .batch(1)
    
    tlab = list(dataset\
    .map(lambda x,lab: lab)\
    .as_numpy_iterator())
    trainy=list(tlab)

    tdat = list(dataset\
    .map(lambda x,lab: x)\
    .as_numpy_iterator())
    trainx=list(tdat)
    
    vallab = list(valDataset\
    .map(lambda x,lab: lab)\
    .as_numpy_iterator())
    valy=list(vallab)

    valdat = list(valDataset\
    .map(lambda x,lab: x)\
    .as_numpy_iterator())
    valx=list(valdat)
    
    labelval = np.vstack(valy)
    labelval = np.argmax(labelval,axis=1)

    xval = np.vstack(valx)
    xval = np.transpose(xval, (0, 2, 1))
    xval_flat = xval.reshape(xval.shape[0],(xval.shape[1]*xval.shape[2]))
    
    labeltr = np.vstack(trainy)
    labeltr = np.argmax(labeltr,axis=1)

    xtr = np.vstack(trainx)
    xtr = np.transpose(xtr, (0, 2, 1))
    xtr_flat = xtr.reshape(xtr.shape[0],(xtr.shape[1]*xtr.shape[2]))
    
    result = (xtr_flat,labeltr,xval_flat,labelval)
    
    return result

def model_rf(params,data):
    
    model = RandomForestClassifier(n_estimators = params[0], max_features = params[1], min_samples_split = params[2], random_state = 42)
    model.fit(data[0], data[1])
    y_pred = model.predict(data[2])
    result = metrics.f1_score(data[3],y_pred)
    
    return result
