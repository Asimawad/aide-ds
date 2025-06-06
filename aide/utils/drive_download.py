import gdown
# https://drive.google.com/file/d/1Jxbvpkor3e7t1T-VW6JyKTlN2SdL1ypS/view?usp=sharing

# https://drive.google.com/file/d/1hQIkbvP6hJ7amIHfv921TjHgiyZhox0a/view?usp=drive_link
file_id = '1Jxbvpkor3e7t1T-VW6JyKTlN2SdL1ypS'
# file_id = '1hQIkbvP6hJ7amIHfv921TjHgiyZhox0a'
output = 'data.zip'  
# output = 'lite_dataset.zip' 

url = f'https://drive.google.com/uc?id={file_id}'
gdown.download(url, output, quiet=False)

# unzip the data.zip file
import zipfile
with zipfile.ZipFile('lite_dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('lite_dataset')
    
# unzip the data.zip file
with zipfile.ZipFile('data.zip', 'r') as zip_ref:
    zip_ref.extractall('data')