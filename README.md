# BIOF399-Final-project

[![Open In Colab]
(https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alechay/BIOF399-Final-project/blob/main/mri_classification.ipynb)

BIOF 399 Deep Learning for Healthcare Image Analysis final project <br>
<br>
Please upload submit an outline for your final project presentation. <br>
Be sure to include title, goal, applications, and expected outcomes.

## Classifiying MRI Data as Demented or Nondemented

The goal of this project is to build a model using DL/TF that can accurately classify MRI Data as demented or nondemented. The data comes from the Open Access Series of Imaging Studies (OASIS) website. I am using the OASIS-2 data, which consists of a longitudinal collection of 150 subjects aged 60 to 96. Each subject was scanned on two or more visits, separated by at least one year for a total of 373 imaging sessions. For each subject, 3 or 4 individual T1-weighted MRI scans obtained in single scan sessions are included. The subjects are all right-handed and include both men and women. 72 of the subjects were characterized as nondemented throughout the study. 64 of the included subjects were characterized as demented at the time of their initial visits and remained so for subsequent scans, including 51 individuals with mild to moderate Alzheimer’s disease. Another 14 subjects were characterized as nondemented at the time of their initial visit and were subsequently characterized as demented at a later visit. For more information, see the article in the repository (Marcus et al. 2010). <br>
Source: OASIS: Longitudinal: Principal Investigators: D. Marcus, R, Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382 (https://www.oasis-brains.org/) 

## Applications

Worldwide, around 50 million people have dementia, with nearly 60% living in low- and middle-income countries. Every year, there are nearly 10 million new cases (https://www.who.int/news-room/fact-sheets/detail/dementia). It is important to be able to accurately diagnose dementia so that people can get the appropriate treatment. However, it is time-intensive and costly to image and diagnose so many people. Machine learning provides a compelling solution to this problem. Automation of dementia diagnosis could save countless hours and dollars, while helping people across the world. The deep learning model I plan to create will certainly have flaws, but I hope it will help to address this problem.

## Expected outcomes

I expect to create a model that performs well above chance in this binary classification task. I will adjust the hyperparameters (dropout, learning rate, etc.), experiment with different layers, change the number of neurons in those layers, and experiment with other modifications to find the best performing model. I will measure the model's performance on the test set using metrics such as precision, recall, and f1-score. I will also evaluate performance through visualizations such as confusion matrices and training accuracy curves. The final product will be the model, saved as an h5 file, and the Jupyter notebook walking the reader through how the model was built and evaluated. 

https://keras.io/examples/vision/3D_image_classification/ <br>
https://www.oasis-brains.org/
