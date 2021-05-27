import math
import ee
from .globals import *

### Harmonic functions for unsupervised anomaly detection
#kernel mean
def addMean(index):
  def addMeanInner(image):
    image = image.addBands(image.select([index]).focal_mean(radius = RADIUS, units = "pixels"))
    return image
  return addMeanInner

# Function to add a time band
def addDependents(image):
  # Compute time in fractional years since the epoch.
  years = ee.Date(image.get('system:time_start')).difference('1970-01-01', 'year')
  timeRadians = ee.Image(years.multiply(2 * math.pi)).rename('t')
  constant = ee.Image(1)
  return image.addBands(constant).addBands(timeRadians.float())


# Function to compute the specified number of harmonics
# and add them as bands.  Assumes the time band is present.
def addHarmonics(freqs):
  def addHarmonicsInner(image):
    # Make an image of frequencies.
    frequencies = ee.Image.constant(freqs)
    # This band should represent time in radians.
    time = ee.Image(image).select('t')
    # Get the cosine terms.
    cosines = time.multiply(frequencies).cos().rename(cosNames)
    # Get the sin terms.
    sines = time.multiply(frequencies).sin().rename(sinNames)
    return image.addBands(cosines).addBands(sines)
  return addHarmonicsInner

#calculate predicted values from model coeffs
#diff predicted and oberved, rescale by RMSE
def diff_predict(hTC,dependents,independents,index,hResid):
  def diff_predict_inner(image):
    image = image\
    .addBands(image.select(independents).multiply(hTC).reduce('sum').rename(index + '_fitted'))
    image = image\
    .addBands(image.select([index]).subtract(image.select([index + '_fitted'])).divide(hResid.select([index + '_resid'])).rename(index+'_diff'))
    return image
  return diff_predict_inner

#add a band of harmonic residuals
def harmon_resid(index,imageCol):
  # The dependent variable we are modeling.
  dependents = ee.List([index])

#change depending on dependents
#independents = ee.List(['constant', index+'_1','t']) \
  independents = ee.List(['constant','t']) \
  .cat(cosNames).cat(sinNames)

  #fit the regression
  # The output of the regression reduction is a 4x1 array image.
  harmonicTrend_tr = imageCol\
  .filterDate(TDATESTART_TR,TDATESTART)\
  .select(independents.cat(dependents))\
  .reduce(ee.Reducer.robustLinearRegression(independents.length(), 1))\

  # Turn the array image into a multi-band image of coefficients.
  hTC = harmonicTrend_tr.select('coefficients').arrayProject([0]).arrayFlatten([independents])
  #RMSE
  hResid = harmonicTrend_tr.select('residuals').arrayFlatten([[index + '_resid']])
  
  #add preductions and residuals
  imageCol = imageCol\
  .filterDate(TDATESTART,TDATEEND)\
  .map(diff_predict(hTC,dependents,independents,index,hResid))

  return imageCol