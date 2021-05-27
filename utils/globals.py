import datetime
import ee
import os
## Files and folders ##

USER_NAME = 'glennwithtwons'
TRAIN_INPUT = 'users/glennwithtwons/reno_ee'
TEST_INPUT = 'users/glennwithtwons/reno_ee_2021'
PREDICT_MASK = 'users/glennwithtwons/remnos'
#PREDICT_MASK = ee.Geometry.Polygon([[[19.639005114855262, -34.289751804879266],[19.639005114855262, -34.31389429181759], [19.66702883464286, -34.31389429181759],[19.66702883464286, -34.289751804879266]]])
TRAIN_ASSET = 'users/glennwithtwons/s2_reno_CNN_dltraining_6'
IM_ASSET = 'users/glennwithtwons/s2_reno_CNN_dl_predict_'
TRAIN_TAB = 'Training_reno_cnn'
TEST_TAB = 'Testing_reno_cnn'
VAL_TAB = 'Valid_reno_cnn'
FILTER_TAB = 'users/glennwithtwons/tffilter'
PREDICT_IMG_BASE = 'Image_reno_predict_'
PREDICT_ASSET_BASE = 'projects/ee-exports/assets/Image_reno_predict_'
PREDICTIONS_IMG = 'Classified_predict.TFRecord'
PREDICTIONS_ASSET = '/Classified_predict'
EXTRACT_POINTS = 'users/glennwithtwons/reno_ee_10'
TEST_POINTS = 'users/glennwithtwons/reno_ee_2021_10'
# REPLACE WITH YOUR BUCKET!
outputBucket = 'ee-tf-example'
outputBucket_predict = 'renosterveld-monitor'

## Save and load ##
assetPath1 =  'users/glennwithtwons/s2_tf'
str_name1 = 'task_phase1_'
assetPath2 =  'users/glennwithtwons/s2_tf_filled'
str_name2 = 'task_phase2_'
assetPath3 =  TRAIN_ASSET + '_flat_' + str(datetime.date.today())
str_name3 = 'renosterveld s2 time series ' + '_' + str(datetime.date.today())
assetPath4 = IM_ASSET + '_flat_' + str(datetime.date.today())
str_name4 = 'renosterveld s2 time series ' + '_' + str(datetime.date.today())

#tf export

# Names for output files.
trainFilePrefix = TRAIN_TAB 
testFilePrefix = TEST_TAB
validFilePrefix = VAL_TAB

fileNameSuffix = '.tfrecord.gz'
trainFilePath = 'gs://' + outputBucket + '/' + trainFilePrefix + fileNameSuffix
testFilePath = 'gs://' + outputBucket + '/' + testFilePrefix + fileNameSuffix
validFilePath = 'gs://' + outputBucket + '/' + validFilePrefix + fileNameSuffix

saveflag = [0,0,0,0]
loadflag = [0,0,0,0]


## Regions ##

#study area
aoi = ee.FeatureCollection(TRAIN_INPUT).geometry().convexHull()
#predict area
poi = ee.FeatureCollection(PREDICT_MASK).geometry().convexHull()

##Parameters##

#percent of points to use
FACTOR = 1
#buffer in meters around polygons to remove edge pixels
BUFFERDIST = 10
#test train split
SPLIT = 0.7
#data thinning split  change
SPLIT2 = 0.33
#data thinning split no change
SPLIT3 = 0.05
#data thinning test split
TEST_SPLIT = 0.33

#cloud prob threshold
CLOUD_THRESH = 40
CLOUD_THRESH_PRE = 50

##harmonics##
# Function to get a sequence of band names for harmonic terms.
def constructBandNames(base, hlist):
  catname = [base + str(x) for x in hlist]
  return ee.List(catname)
# The number of cycles per year to model.
harmonics = 1
# Make a list of harmonic frequencies to model.
# These also serve as band name suffixes.
harmonicFrequencies = ee.List.sequence(1, harmonics)
harmonicFrequencies_st = list(range(1,harmonics+1))
# Construct lists of names for the harmonic terms.
cosNames = constructBandNames('cos_', harmonicFrequencies_st)
sinNames = constructBandNames('sin_', harmonicFrequencies_st)
# Independent variables = intercept, time, and kernel mean
independents = ee.List(['constant','t']) \
  .cat(cosNames).cat(sinNames)

#unsupervised limits
RMSE_LIM = 3
CON_LIM = 3

##Dates##
def make_tdatelist(n):
  return Date_Start.advance(n,'day')

def make_testdatelist(n):
  return TDate_Start.advance(n,'day')

def make_pdatelist(n):
  return PDate_Start.advance(n,'day')

#window length in days
DWINDOW = 180
#window length in steps
dstep = 10
WINDOW = int(DWINDOW/dstep)

#create dates sequence
#train/val
DATESTART = ee.Date('2015-12-09')
DATEEND = ee.Date('2019-12-31')
DATEENDpy = datetime.datetime.strptime('2019-12-31', '%Y-%m-%d')

Date_Start = ee.Date(DATESTART)
Date_End = ee.Date(DATEEND)
n_day = Date_End.difference(Date_Start,'day').round()
dates = ee.List.sequence(0,n_day,dstep)
dates = dates.map(make_tdatelist)
#predictioon dates

#predict
PDATESTR = datetime.datetime.today().strftime('%Y-%m-%d')
PDATEEND = ee.Date(PDATESTR).advance(-1,'day')
PREDICT_IMG = PREDICT_IMG_BASE + "_" + PDATESTR
PREDICT_ASS = PREDICT_ASSET_BASE  + "_" + PDATESTR
PDATESTART = PDATEEND.advance(-1*DWINDOW,'day')

PDate_Start = ee.Date(PDATESTART)
PDate_End = ee.Date(PDATEEND)
n_day_pred = DWINDOW-dstep
pdates = ee.List.sequence(0,n_day_pred,dstep)
pdates = pdates.map(make_pdatelist)

#test
TDATESTART_TR = DATESTART
TDATESTART = ee.Date('2019-06-01')
TDATEEND = ee.Date('2020-10-01')
TDATEENDpy = datetime.datetime.strptime('2020-10-01', '%Y-%m-%d')

TDate_Start = ee.Date(TDATESTART_TR)
TDate_End = ee.Date(TDATEEND)
t_n_day = TDate_End.difference(TDate_Start,'day').round()
t_dates = ee.List.sequence(0,t_n_day,dstep)
t_dates = t_dates.map(make_testdatelist)
## Features ##

#define labels and features
label = 'label'
# Use these bands for prediction.
bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6','B7', 'B8', 'B8A', 'B9', 'B10', 'B11','B12','ndvi','evi','ndre','ndwi','nbr']
#harmonic result
hbands = ['ndvi_diff']
#combined
allbands = bands
#manual test names
testNames = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','ndvi','evi','ndre','ndwi','nbr','ndvi_diff','label','count','roll']

# This is list of all the properties we want to export.
featureNames = list(allbands)
featureNames.append(label)

#TF
#classes
NCLASS = 2
LABELS = [0.0,1.0,2.0]
LABELS_FILTER = [0.0,1.0]
buff = 300000
batch_size=128
n_classes=NCLASS
batch_size=128
epochs=20
steps = 500
input_shape = (None,WINDOW,len(allbands))
tsLength= 149
testLength = 49
lr = 0.0001

model_directory = 'data/test_model'
checkpoints_path = os.path.join(model_directory, 'checkpoints', 'model.ckpt')