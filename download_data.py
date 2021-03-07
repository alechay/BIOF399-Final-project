import os
import shutil
import urllib.request
import tarfile

# create data directory and download files
if os.path.exists("data") == False:
    os.mkdir("data")

    url1 = 'https://download.nrg.wustl.edu/data/OAS2_RAW_PART1.tar.gz'
    url2 = 'https://download.nrg.wustl.edu/data/OAS2_RAW_PART2.tar.gz'
    url3 = 'https://www.oasis-brains.org/files/oasis_longitudinal_demographics.xlsx'
    
    print('Beginning MRI file 1 download...')
    urllib.request.urlretrieve(url1, 'data/OAS2_RAW_PART1.tar.gz')
    print('Beginning MRI file 2 download...')
    urllib.request.urlretrieve(url2, 'data/OAS2_RAW_PART2.tar.gz')
    print('Beginning Excel file download...')
    urllib.request.urlretrieve(url3, 'data/oasis_longitudinal_demographics.xlsx')
    
# extract tar.gz files and remove them
if os.path.exists("data/OAS2_RAW_PART1.tar.gz"):
    print("Extracting MRI file 1...")
    tar1 = tarfile.open('data/OAS2_RAW_PART1.tar.gz', "r:gz")
    tar1.extractall('data')
    tar1.close()
    os.remove("data/OAS2_RAW_PART1.tar.gz")

if os.path.exists("data/OAS2_RAW_PART2.tar.gz"):
    print("Extracting MRI file 2...")
    tar2 = tarfile.open('data/OAS2_RAW_PART2.tar.gz', "r:gz")
    tar2.extractall('data')
    tar2.close()
    os.remove("data/OAS2_RAW_PART2.tar.gz")

# make OAS2_RAW directory
if os.path.exists("data/OAS2_RAW") == False:
    print("Copying files...")
    os.mkdir("data/OAS2_RAW")
    
# move files from PART1 and PART2 to OAS2_RAW and remove empty directory
if os.path.exists("data/OAS2_RAW_PART1"):
    file_names = os.listdir('data/OAS2_RAW_PART1')
    for file_name in file_names:
        shutil.move(os.path.join('data/OAS2_RAW_PART1', file_name), 'data/OAS2_RAW')
    os.rmdir("data/OAS2_RAW_PART1")
    
if os.path.exists("data/OAS2_RAW_PART2"):
    file_names = os.listdir('data/OAS2_RAW_PART2')
    for file_name in file_names:
        shutil.move(os.path.join('data/OAS2_RAW_PART2', file_name), 'data/OAS2_RAW')
    os.rmdir("data/OAS2_RAW_PART2")
    
print("Done!")