import datetime
import ee
from .globals import *

## Misc utility functions

#to fill nulls with previous non null record
def gap_fill(image, listb):
  previous = ee.Image(ee.List(listb).get(-1))
  new = image.unmask(previous)
  return ee.List(listb).add(new)

#roll count
def roll_count(image, listb):
  previous = ee.Image(ee.List(listb).get(-1))
  count = previous.select('count')
  roll = previous.select('roll')

  #if error is > 3 add to count
  count = count.where(image.select(hbands).abs().gte(ee.Image(RMSE_LIM)),count.add(ee.Image(1)))
  #if error is < 3 reset count
  count = count.where(image.select(hbands).abs().lt(ee.Image(RMSE_LIM)),ee.Image(0)).rename('count')
  #land cover change flag
  roll = roll.where(count.gte(CON_LIM),ee.Image(1))
  
  count= count.toInt()
  roll=roll.toInt()
  
  out = image.addBands(count).addBands(roll)

  return ee.List(listb).add(out)

#get min/max to standardize
def eeminmax(bound,Lim,labelint):
  Pim = Lim.reduceRegion(\
    reducer= ee.Reducer.percentile([bound]),\
    geometry=  aoi,\
    scale= 30,\
    maxPixels= 10e9,\
    tileScale=8)\
    .set(label,labelint)\
    .getInfo()

  return Pim

#function to mask renoster pathces
def remclip(mask):
    def remclipInner(image):
      return image.updateMask(mask)
    return remclipInner

#function to remove obs ndvi < 0
def ndclip(image):
  return image.updateMask(image.select(['ndvi']).gt(0))
  
#convert numeric feature to date
def num2date(feature):
  d1 = ee.Date(feature.get('date_1'))
  d2 = ee.Date(feature.get('date_2'))
  return feature.set('date_1', d1).set('date_2', d2)

def dateset(image):
  date = ee.Date(image.get('system:index'))
  return image.set('system:time_start',date)
  
##save assets to prevent OOM errors
def saveAssets(imcoll,assetPath,length,str_name,dlist):
  ilist = imcoll.toList(length)
  for idx in range(0, length):

          #convert to img and set time
          img = ilist.get(idx)
          time1 = dlist[idx]
          img = ee.Image(img)
          name = time1
          description = str_name + str(idx)

          assetId = assetPath+"/"+name

          task = ee.batch.Export.image.toAsset(image=img,
                                               assetId=assetId,
                                               description=description,
                                               region=aoi,
                                               maxPixels=200000000,
                                               scale=10)
          task.start()
          print('Exporting {} to {}'.format(name, assetId))

  return None