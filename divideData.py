import os
import shutil
import random
from PIL import Image

def is_image_ok(fn):
    try:
        Image.open(fn)
        return True
    except:
        return False

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def get_files(mypath):
    return [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

def recreateFolder(dividedDataPath):
    if(os.path.isdir(dividedDataPath)):
        tempPath = os.path.join(os.path.split(dividedDataPath)[0], "temp")
        shutil.move(dividedDataPath, tempPath) #done to prevent error on mkdir
        shutil.rmtree(tempPath, ignore_errors=True)#, ignore_errors=False, onerror=None)
    os.mkdir(dividedDataPath)
    
    
def DivideData(originalDataPath, dividedDataPath, validatePercent, testPercent, batchSize = 1):
    trainPercent = 100.0 - validatePercent - testPercent

    #clear and create the target folder
    recreateFolder(dividedDataPath)
    
    subdirs = ["test","validation","train"]
    subsubdirs = get_immediate_subdirectories(originalDataPath)

    #create test, validation and train folders
    for subdir in subdirs:
        newdir = os.path.join(dividedDataPath, subdir)
        os.mkdir(newdir)
        for subsubdir in subsubdirs:
            newsubdir = os.path.join(newdir, subsubdir)
            os.mkdir(newsubdir)
    
    #assigning the files to test/validate/train
    nb_files = 0
    for subdir in subsubdirs:
        originalDir = os.path.join(originalDataPath, subdir)
        nb_files += len(get_files(originalDir))
    nb_files = (nb_files//batchSize)*batchSize

    for subdir in subsubdirs:
        originalDir = os.path.join(originalDataPath, subdir)  
        test_dir = os.path.join(dividedDataPath, subdirs[0], subdir)
        validdir = os.path.join(dividedDataPath, subdirs[1], subdir)
        traindir = os.path.join(dividedDataPath, subdirs[2], subdir)
        
        files = get_files(originalDir)
        random.shuffle(files)

        noOfFiles = int(min([len(files),nb_files]))
        nb_files -= noOfFiles
        
        noOfTest = int(noOfFiles*(testPercent/100))
        noOfTest = (noOfTest//batchSize)*batchSize
        
        noOfVali = int(noOfFiles*(validatePercent/100))
        noOfVali = (noOfVali//batchSize)*batchSize
        
        noOfTrain = int(noOfFiles*(trainPercent/100))
        noOfTrain = (noOfTrain//batchSize)*batchSize
        
        targets = [(test_dir,noOfTest),(validdir,noOfVali),(traindir,noOfTrain)]
        for (target, count) in targets:
            for i in range(count):
                file = files.pop()
                fileCurrentPath = os.path.join(originalDir, file)
                fileTarget = os.path.join(target, file)
                os.link(fileCurrentPath, fileTarget)

def readCategories(dividedDataPath):
    subdirs = ["test","validation","train"]
    categories = get_immediate_subdirectories(os.path.join(dividedDataPath, subdirs[0]))   
    
    ret = {}
    
    for categori in categories:
        cat = ()
        for subdir in subdirs:
            dir = os.path.join(dividedDataPath, subdir, categori)
            noOfFiles = len(get_files(dir))
            cat += (noOfFiles,)
        ret[categori] = cat
    
    return ret #[("categori1", 1, 2, 3), ("categori2", 1, 2, 3)] #Categori = "eksem"/"psoriasis" , 1 = antal filer i test, 2 = antal filer i validate, 3 = antal filer i train

def getRandomSubset(_from, _to, noOfFiles):
    files = get_files(_from)
    if len(files) < noOfFiles:
        raise InputError("not enough files in from folder")
    
    random.shuffle(files)
    
    for i in range(noOfFiles):
        file = files.pop()
        fileCurrentPath = os.path.join(_from, file)
        fileTarget = os.path.join(_to, file)
        os.link(fileCurrentPath, fileTarget)
        
def deleteContent(deleteFolder):
    for del_file in get_files(deleteFolder):
        del_path = os.path.join(deleteFolder, del_file)
        try:
            os.unlink(del_path)
        except Exception as e:
            print(e)
    
        
def doRandomRequest(path_from,path_to,path_to_predict):
    recreateFolder(path_to_predict)
    #deleteContent(path_to_predict)
    
    #remove and create "testImages"
    recreateFolder(path_to)
    
    #Create Subfolders
    subfolders = get_immediate_subdirectories(path_from)
    
    #create subfolders and fill with data
    noOfFiles = 10
    for subfolder in subfolders:
        newdir = os.path.join(path_to, subfolder)
        os.mkdir(newdir)
        
        oldDir = os.path.join(path_from, subfolder)
        getRandomSubset(oldDir, newdir, noOfFiles)
    
        
def exileCorruptPictures(directory):
    exile = os.path.join(directory, "corrupt")
    
    if not os.path.exists(exile):
        os.makedirs(exile)

    files = get_files(directory)
    
    counter = 0
    for file in files:
        filePath = os.path.join(directory, file)
        if not is_image_ok(filePath):
            counter += 1
            exilePath = os.path.join(exile, file)
            os.rename(filePath, exilePath)
        
    print(str(counter) + " corrupt files found in " + directory)
