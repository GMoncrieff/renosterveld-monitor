from geetools import cloud_mask, batch
import math
import ee
from .globals import *
from .misc_ee import *

#join s2 clouds and s2 data
def joinS2(area_to_join,start,end):
  innerJoin = ee.Join.inner()

  # Specify an equals filter for image timestamps.
  filterIDEq = ee.Filter.equals(
    leftField= 'system:index',
    rightField= 'system:index'
  )

  #level 1 s2 data
  S2_nocloud = ee.ImageCollection('COPERNICUS/S2')\
  .filterBounds(area_to_join)\
  .filterDate(start, end)
  #s2cloudless data
  S2_cloud = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")\
  .filterBounds(area_to_join)\
  .filterDate(start, end)

  innerJoinedS2 = innerJoin.apply(S2_nocloud, S2_cloud, filterIDEq)

  #Map a function to merge the results in the output FeatureCollection
  joinedS2 = innerJoinedS2.map(lambda feature: ee.Image.cat(feature.get('primary'), feature.get('secondary')))
  return ee.ImageCollection(joinedS2)

#return S2 data in regular time series
def getS2(renoster,mask_renoster):
    def getS2inner(dlist):
      #extract dates
      #####
      #start day
      dat = ee.Date(dlist)
      #end day
      dat2 = dat.advance(-1*dstep,'day')
      #x months before (window length)
      dat3 = dat.advance(-1*dstep*WINDOW,'day')
      #string
      dat_str = dat.format()

      #filter renoster and rasterize
      #####
      #still renosterveld if last renosterveld day 'date_1' is after end day
      reno1 = renoster\
      .filter(ee.Filter.gte('date_1', dat))\
      .map(lambda feature: feature.set('class',  ee.Number(0)))\
      .reduceToImage(['class'],ee.Reducer.first())

      #unknown if last renosterveld day 'date_1' is before end day but first transformed day 'date_2' is still after start day
      reno2 = renoster\
      .filter(ee.Filter.And(ee.Filter.gt('date_2', dat2),ee.Filter.lt('date_1', dat)))\
      .map(lambda feature: feature.set('class',  ee.Number(10)))\
      .reduceToImage(['class'],ee.Reducer.first())

      #transformed if first transformed day 'date_2' is before end day, but last renosterveld day is still in window
      reno3 = renoster\
      .filter(ee.Filter.And(ee.Filter.lte('date_2', dat2),ee.Filter.gte('date_1', dat3)))\
      .map(lambda feature: feature.set('class',  ee.Number(1)))\
      .reduceToImage(['class'],ee.Reducer.first())

      #unknown if first transformed day 'date_2' is after winbdow, but last renosterveld day is before
      reno4 = renoster\
      .filter(ee.Filter.And(ee.Filter.gt('date_2', dat3),ee.Filter.lt('date_1', dat3)))\
      .map(lambda feature: feature.set('class',  ee.Number(100)))\
      .reduceToImage(['class'],ee.Reducer.first())

      #stable transformed start day is (WINDOW) days after first transformed day
      reno5 = renoster\
      .filter(ee.Filter.lt('date_2', dat3))\
      .map(lambda feature: feature.set('class',  ee.Number(2)))\
      .reduceToImage(['class'],ee.Reducer.first())

      #combine
      reno_all = reno1.addBands(reno2).addBands(reno3).addBands(reno4).addBands(reno5)
      reno_all = reno_all.reduce('sum')
      reno_all = reno_all.select('sum').rename('label')

      #join
      S2_joined = joinS2(aoi,dat2,dat)

      #process joined data
      imageCol_gaps = S2_joined\
      .filterBounds(aoi)\
      .filterDate(dat2, dat)\
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESH_PRE))\
      .map(shadowCloudMask)\
      .map(lambda image: image.divide(10000))\
      .map(remclip(mask_renoster))\
      .map(addBANDS)\
      .select(bands)
      #.map(lambda image: image.updateMask(image.select(['probability']).lt(CLOUD_THRESH)))\

      #need to deal with cases in which we have no image
      imageCol_gaps = ee.Algorithms.If(imageCol_gaps.size(),\
                                      imageCol_gaps.qualityMosaic('ndvi'),\
                                       ee.ImageCollection(ee.List(bands).map(lambda band: ee.Image())).toBands().rename(bands))
           
      imageCol_gaps = ee.Image(imageCol_gaps).updateMask(mask_renoster)

      #remove when ndvi <0
      imageCol_gaps = ndclip(imageCol_gaps)
      #add labels and time
      imageCol_gaps = imageCol_gaps\
      .addBands(reno_all)\
      .set('system:time_start', dat)

      return imageCol_gaps
    return getS2inner

#return S2 data in regular time series for prediction
def predS2(renoster,mask_renoster):
    def predS2inner(dlist):
      #extract dates
      #start day
      dat = ee.Date(dlist)
      #end day
      dat2 = dat.advance(-1*dstep,'day')
      #string
      dat_str = dat.format()

      #join
      S2_joined = joinS2(poi,dat2,dat)

      #process joined data
      imageCol_gaps = S2_joined\
      .filterBounds(poi)\
      .filterDate(dat2, dat)\
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))\
      .map(shadowCloudMask)\
      .map(lambda image: image.divide(10000))\
      .map(remclip(mask_renoster))\
      .map(addBANDS)\
      .select(bands)
      #.map(lambda image: image.updateMask(image.select(['probability']).lt(CLOUD_THRESH)))\

      #need to deal with cases in which we have no image
      imageCol_gaps = ee.Algorithms.If(imageCol_gaps.size(),\
                                      imageCol_gaps.qualityMosaic('ndvi'),\
                                       ee.ImageCollection(ee.List(bands).map(lambda band: ee.Image())).toBands().rename(bands))
      imageCol_gaps = ee.Image(imageCol_gaps).updateMask(mask_renoster)

      #remove when ndvi <0
      imageCol_gaps = ndclip(imageCol_gaps)

      #add labels and time
      imageCol_gaps = imageCol_gaps\
      .set('system:time_start', dat)

      return imageCol_gaps
    return predS2inner
  
#S2 shadows and clouds from https://github.com/samsammurphy/cloud-masking-sentinel2/blob/master/cloud-masking-sentinel2.ipynb
def shadowCloudMask(image):
    """
    Finds cloud shadows in images
    
    Originally by Gennadii Donchyts, adapted by Ian Housman
    """
    
    def potentialShadow(cloudHeight):
        """
        Finds potential shadow areas from array of cloud heights
        
        returns an image stack (i.e. list of images) 
        """
        cloudHeight = ee.Number(cloudHeight)
        
        # shadow vector length
        shadowVector = zenith.tan().multiply(cloudHeight)
        
        # x and y components of shadow vector length
        x = azimuth.cos().multiply(shadowVector).divide(nominalScale).round()
        y = azimuth.sin().multiply(shadowVector).divide(nominalScale).round()
        
        # affine translation of clouds
        cloudShift = cloudMask.changeProj(cloudMask.projection(), cloudMask.projection().translate(x, y)) # could incorporate shadow stretch?
        
        return cloudShift

    # select a cloud mask
    cloudMask = image.select(['probability'])

    img = image.select(['B1','B2','B3','B4','B6','B8A','B9','B10', 'B11','B12'],\
                 ['aerosol', 'blue', 'green', 'red', 'red2','red4','h2o', 'cirrus','swir1', 'swir2'])\
                 .divide(10000).addBands(image.select('QA60'))\
                 .set('solar_azimuth',image.get('MEAN_SOLAR_AZIMUTH_ANGLE'))\
                 .set('solar_zenith',image.get('MEAN_SOLAR_ZENITH_ANGLE'))
    
    # make sure it is binary (i.e. apply threshold to cloud score)
    cloudScoreThreshold = CLOUD_THRESH
    cloudMask = cloudMask.gt(cloudScoreThreshold)

    # solar geometry (radians)
    azimuth = ee.Number(img.get('solar_azimuth')).multiply(math.pi).divide(180.0).add(ee.Number(0.5).multiply(math.pi))
    zenith  = ee.Number(0.5).multiply(math.pi ).subtract(ee.Number(img.get('solar_zenith')).multiply(math.pi).divide(180.0))

    # find potential shadow areas based on cloud and solar geometry
    nominalScale = cloudMask.projection().nominalScale()
    cloudHeights = ee.List.sequence(500,4000,500)        
    potentialShadowStack = cloudHeights.map(potentialShadow)
    potentialShadow = ee.ImageCollection.fromImages(potentialShadowStack).max()

    # shadows are not clouds
    potentialShadow = potentialShadow.And(cloudMask.Not())

    # (modified) dark pixel detection 
    darkPixels = img.normalizedDifference(['green', 'swir2']).gt(0.25)

    # shadows are dark
    shadows = potentialShadow.And(darkPixels)
    cloudShadowMask = shadows.Or(cloudMask)

    return image.updateMask(cloudShadowMask.Not())


# Function to mask clouds using the Sentinel-2 QA band.
def maskS2_L1_clouds(image):
  qa = image.select('QA60')

  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloudBitMask = ee.Number(2).pow(10).int()
  cirrusBitMask = ee.Number(2).pow(11).int()

  # Both flags should be set to zero, indicating clear conditions.
  mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
             qa.bitwiseAnd(cirrusBitMask).eq(0))

  # Return the masked and scaled data.
  return image.updateMask(mask).divide(10000)

#hollstein mask
#from https://www.mdpi.com/2072-4292/8/8/666/pdf
def mask_holl(image):
  image = cloud_mask.applyHollstein(image)

  # Return the masked and scaled data.
  return image


#add bands
def addNDVI(image):
  return image.addBands(image.normalizedDifference(['B8', 'B4']).rename(['ndvi']))

def addEVI(image):
  return image.addBands(image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
      'NIR': image.select('B8'),
      'RED': image.select('B4'),
      'BLUE': image.select('B2')
    }).rename(['evi']))

def addNDRE(image):
  return image.addBands(image.normalizedDifference(['B8', 'B5']).rename(['ndre']))

def addNDWI(image):
  return image.addBands(image.normalizedDifference(['B8', 'B11']).rename(['ndwi']))

def addNBR(image):
  return image.addBands(image.normalizedDifference(['B8', 'B12']).rename(['nbr']))

#all bands
def addBANDS(image):
  img = image.addBands(image.normalizedDifference(['B8', 'B4']).rename(['ndvi']))\
  .addBands(image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
      'NIR': image.select('B8'),
      'RED': image.select('B4'),
      'BLUE': image.select('B2')
    }).rename(['evi']))\
  .addBands(image.normalizedDifference(['B8', 'B5']).rename(['ndre']))\
  .addBands(image.normalizedDifference(['B8', 'B11']).rename(['ndwi']))\
  .addBands(image.normalizedDifference(['B8', 'B12']).rename(['nbr']))
  
  return img

