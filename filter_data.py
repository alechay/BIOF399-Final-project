import os
import shutil
from zipfile import ZipFile

import pandas as pd

if os.path.exists("filtered_data") == False:
    
    os.mkdir('filtered_data')
    os.mkdir('filtered_data/nondemented')
    os.mkdir('filtered_data/demented')

    # read demographics info
    df = pd.read_excel('data/oasis_longitudinal_demographics.xlsx')

    # Exclude where group is 'Converted'. The goal is just to classify demented vs nondemented images.
    subjects = df[(df['Group']!='Converted')]
    subjects = subjects[['Subject ID', 'MRI ID', 'Group']]

    # get lists of all mri images and mri id's
    images = []
    mri_id = []
    scan_num = []

    for s in subjects['MRI ID']:
        d = os.path.join('data/OAS2_RAW', s, 'RAW')

        for file in os.listdir(d):
            scan_num.append(os.path.splitext(file)[0]) # get scan number

            file = os.path.join(d, file) # get full path name
            images.append(file) # load file with nibabel
            mri_id.append(s)      

    # if it has worked properly, the length of the lists should be the same
    assert(len(images) == len(mri_id))

    # create image df
    image_df = pd.DataFrame(list(zip(mri_id, scan_num, images)), columns =['MRI ID', 'Scan number', 'Image'])

    # merge the two df's
    subjects = subjects.merge(image_df, how='right', on='MRI ID')
    # there should be no null values in the Image column
    assert(subjects['Image'].isnull().values.any()==False)
    
    # filter to just include first scan on first visit
    filtered = subjects[(subjects['MRI ID'].str.endswith('MR1')) &
        (subjects['Scan number'] == 'mpr-1.nifti')]
    
    # copy files to new directory
    for index, row in filtered.iterrows():
        newname = row['Subject ID'] + '_' + os.path.basename(row['Image'])
        if row['Group'] == 'Nondemented':
            newpath = os.path.join('filtered_data/nondemented', newname)
            shutil.copy(row['Image'], newpath)
        else:
            newpath = os.path.join('filtered_data/demented', newname)
            shutil.copy(row['Image'], newpath)
    
else:
    print('Filtered data already exists!')