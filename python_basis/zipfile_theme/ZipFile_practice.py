from zipfile import ZipFile
import os
import shutil
from fnmatch import fnmatch, fnmatchcase
DL_DIR_PREFIX = "model/model"
with ZipFile("./models/model.zip",'r') as f:
    zips = list(filter(lambda x: x.startswith(DL_DIR_PREFIX), f.namelist()))
    print(zips)
    f.extractall("./models", members=zips)
