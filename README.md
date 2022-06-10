# Shrubland change detection using neural networks
_Code for the paper 'Continuous Land Cover Change Detection in a Critically Endangered Shrubland Ecosystem Using Neural Networks' by Glenn Moncrieff_  

[https://www.mdpi.com/2072-4292/14/12/2766/htm](https://www.mdpi.com/2072-4292/14/12/2766/htm)

Google Earth Engine is required to run notebooks. Code was run on a GCP Vertex AI Workbench VM.

[01_data_export.ipynb](https://github.com/GMoncrieff/renosterveld-monitor/blob/main/01_data_export.ipynb) exports train, test andn valid data to google cloud storage
[02_model_fit.ipynb](https://github.com/GMoncrieff/renosterveld-monitor/blob/main/02_model_fit.ipynb) fits models using tf2 with preselected parameters values
[03_predict.ipynb](https://github.com/GMoncrieff/renosterveld-monitor/blob/main/03_predict.ipynb) uses saved model to predict for a specific date over a region and upload results to earth engine for visualization
[04_salient.ipynb](https://github.com/GMoncrieff/renosterveld-monitor/blob/main/04_salient.ipynb) calculates saliency using grad-CAM++ on temp-CNN model

Global variables defining region, dates, parameters, filenames, credentials etc are defined in [utils/globals.py](https://github.com/GMoncrieff/renosterveld-monitor/blob/main/utils/globals.py)

Code for the operational prediction pipeline implemented using google cloud functions, cloud run and cloud dataflow can be found at [https://github.com/mgietzmann/global_renosterveld_watch](https://github.com/mgietzmann/global_renosterveld_watch). This prediction pipeline makes predicitons of land cover change and uploads them to Earth Engine every 20 days. Results can be viewed at [https://glennwithtwons.users.earthengine.app/view/global-renosterveld-watch](https://glennwithtwons.users.earthengine.app/view/global-renosterveld-watch)



