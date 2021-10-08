"""
databuilder.py
Builds the data needed by the test models
Run this file first to auto generate all the trips matrices and raw data needed to run the models.
Look in globals.py for the names of all the files it creates.

NOTE: the "zonecodes.csv" file is central to the code and contains a lookup between the geographic
area code (e.g. "E02000001") and the zone number (e.g. "0") which is the row and col number in all
the matrices.

There were a number of different data sources that were used to create the training data for the
GISTAM 2019 paper. The best option is to use:

quant1InstallFromTrainingData(tmpDataDir)

which downloads three gigabyte files from osf.io and unpacks them into pickled matrices ready
for use by Python. See the main program at the bottom of this listing.

The alternative:

quant1FullInstall(tmpDataDir)

does the same thing, but using raw data. This works as follows:

1. Download trips data from wu03ew_v2.csv if it's not already here
https://www.nomisweb.co.uk/output/census/2011/wu03ew_msoa.zip
2. Make this raw trips data into matrices for road, bus and rail as Python pickle dumps
3. Download the QUANT1 matrices to match the GISTAM 2019 paper from osf.io as C# binary
files which are converted to Python pickle. There is no easy way of doing this directly
as it relies on the entire road network for England and Wales, along with the bus and
rail timetables, to create zone to zone travel times in minutes for every zone pair.
That's 7201x7201=52 million trip times on million node networks. On a high end GPU,
you're looking at around 3 hours processing for road, 1 hour for bus and 5 minutes for rail.
So we download ready built trip time matrices, which is fine for this experiment.

Then, the QUANT2 installer is an update to the GISTAM 2019 paper, as QUANT2 now adds
data for Scotland to make an integrated Great Britain model where QUANT1 was just
England and Wales. All the trips and cost matrices are essentially identical, it's just
that there are more areas.

All configuration for what to install is in the main program at the end of this
listing.

The 'tmpDataDir' is where temporary files are downloaded to that can be deleted after
installation. The 'modelRunsDir' contains the live files which are needed to run models.
"""

import os
import os.path
import zipfile
import urllib.request
import ssl
#THIS IS VERY BAD - urllib request throws bad cert on OSF.io otherwise
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
from shutil import copyfile
from globals import *
#from utils import loadZoneLookup #moved to zonecodes
from zonecodes import ZoneCodes
from utils import generateTripsMatrix
from utils import loadMatrix, saveMatrix, loadQUANTMatrix, loadQUANTCSV


#definitions of where everything downloads from and goes to

tmpDataDir = 'data' #tmpDataDir contains data downloaded or created from QUANT used to make training sets
modelRunsDir = 'model-runs' #where the 'live' data needed to run models goes

urlWU03EU = 'https://www.nomisweb.co.uk/output/census/2011/wu03ew_msoa.zip'
#QUANT1 matrices for GISTAM 2019 are available here (England/Wales only 7201 zones):
#https://osf.io/bgcen/ for the project containing the QUANT1 data
url_QUANT1_trainingdata_Road_csv = 'https://osf.io/8vcjq/download' #this is origin, dest, TObsFlow, Cij data
url_QUANT1_trainingdata_Bus_csv = 'https://osf.io/abjqm/download'
url_QUANT1_trainingdata_GBRail_csv = 'https://osf.io/fvqn4/download'
url_QUANT1_TijRoad_csv = 'https://osf.io/sp4qd/download' #TijRoadObs.csv
url_QUANT1_TijBus_csv = 'https://osf.io/7ujrz/download' #TijBusObs.csv
url_QUANT1_TijGBRail_csv = 'https://osf.io/kdt5b/download' #TijRailObs.csv
url_QUANT1_TijRoad_qcs = 'https://osf.io/65zy9/download' #TObs_1.bin (QUANT C Sharp)
url_QUANT1_TijBus_qcs = 'https://osf.io/e4hy7/download' #TObs_2.bin (QUNAT C Sharp)
url_QUANT1_TijGBRail_qcs = 'https://osf.io/e6fm5/download' #TObs_3.bin (QUANT C Sharp)
url_QUANT1_CijRoadMin_csv = 'https://osf.io/4fhp3/download' #dijRoad_min.csv
url_QUANT1_CijBusMin_csv = 'https://osf.io/wjmsa/download' #dijBus_min.csv
url_QUANT1_CijRailMin_csv = 'https://osf.io/rtme9/download' #dijRail_min.csv
url_QUANT1_CijRoadMin_qcs = 'https://osf.io/7jpvm/download' #dis_roads_min.bin (QUANT C Sharp)
url_QUANT1_CijBusMin_qcs = 'https://osf.io/7wqf3/download' #dis_bus_min.bin (QUANT C Sharp)
url_QUANT1_CijGBRailMin_qcs = 'https://osf.io/4m6ug/download' #dis_gbrail_min.bin (QUANT C Sharp)
url_QUANT1_CijCrowflyKM_qcs = 'https://osf.io/5cjyn/download' #dis_crowfly_KM.bin (QUANT C Sharp)
url_QUANT1_ZoneCodesText = 'https://osf.io/hu9zw/download' #ZoneCodesText.csv

#QUANT2 matrices, from OSF data site, which now include Scotland (8436 zones):
#url_QUANT2_EWS_censustraveltowork = 'https://osf.io/du8ar/download' #makes Tij matrices
url_QUANT2_trainingdata_Road_csv = '' #this is origin, dest, TObsFlow, Cij data
url_QUANT2_trainingdata_Bus_csv = ''
url_QUANT2_trainingdata_GBRail_csv = ''
url_QUANT2_TijRoad_qcs = 'https://osf.io/ga9m3/download'
url_QUANT2_TijBus_qcs = 'https://osf.io/nfepz/download'
url_QUANT2_TijRail_qcs = 'https://osf.io/at9vc/download'
url_QUANT2_CijRoadMin_qcs = 'https://osf.io/u2mz6/download'
url_QUANT2_CijBusMin_qcs = 'https://osf.io/bd4s2/download'
url_QUANT2_CijGBRailMin_qcs = 'https://osf.io/gq8z7/download'
url_QUANT2_ZoneCodesText = 'https://osf.io/hu7bd/download'

"""
Utility to check if files exist and download QUANT format C Sharp binary files and convert
to pickle for python. The QUANT matrices are just m,n dimensions as two 4 byte IEEE754 floats
then m x n 4 byte floats containing all the row and column data.
@param localFilename This is the file that we're going to create - this is the pickled one
@param localDir This is the dir that contains localFilename
@param url This is the url to download it from if above localFilename doesn't exist
@param qcsFilename This is the filename of the C Sharp binary file downloaded from the url
which may be the same as the localFilename, or could be different.
"""
def ensureMatrixFileQUANTtoPickle(localFilename, localDir, url, qcsFilename):
    if os.path.isfile(os.path.join(localDir, localFilename)):
        print('databuilder.py:',localFilename,' exists - skipping')
    else:
        print('databuilder.py: ',localFilename,' downloading from ',url)
        path_qcs = os.path.join(localDir,qcsFilename)
        urllib.request.urlretrieve(url, path_qcs)
        #load the QUANT CSharp format matrix and pickle it for python
        M = loadQUANTMatrix(path_qcs)
        path_pic = os.path.join(localDir,localFilename)
        print('databuilder.py: saving file ',path_pic)
        saveMatrix(M,path_pic) #save pickled matrix
###

"""
Utility to check for the existence of a plain file and download it from the given url if
it does not exist.
@param localFilename the name of the file in the localDir dir
@param localDir the dir containing localFilename
@param the url to download it from if not present on the current file system
"""
def ensurePlainFile(localFilename, localDir, url):
    if os.path.isfile(os.path.join(localDir, localFilename)):
        print('databuilder.py:',localFilename,' exists - skipping')
    else:
        print('databuilder.py: ',localFilename,' downloading from ',url)
        path = os.path.join(localDir,localFilename)
        urllib.request.urlretrieve(url, path)
        print('databuilder.py: created file ',localFilename,' in',localDir)
###

"""
Check if localFilename exists in the dir directory. If not, then download
it (as a zip file) from the given url and unzip it in dir.
@param localFilename The file we're checking for its existence
@param localDir The directory containing localFilename
@param url The url to download localFilename from if it doesn't exist. Download goes into
dir/localFilename and then gets unzipped into the same directory.
"""
def ensurePlainZIPFile(localFilename, localDir, url):
    if os.path.isfile(os.path.join(localDir, localFilename)):
        print('databuilder.py:',localFilename,' exists - skipping')
    else:
        #NO! need to download a zip!
        print('databuilder.py: ',localFilename,' downloading from ',url)
        basename, ext = os.path.splitext(localFilename) #what we download is a .zip file
        zippath = os.path.join(localDir,basename+'.zip')
        urllib.request.urlretrieve(url, zippath)
        print('databuilder.py: created file ',localFilename,' in ',localDir)
        #now unzip it
        zip_ref = zipfile.ZipFile(zippath, 'r')
        zip_ref.extractall(localDir)
        zip_ref.close()
###

################################################################################
# Now on to the data production
################################################################################

"""
DEPRECATED
Load data from a csv format where the csv just contains rows and columns of data
for a matrix. This can be loaded directly into a numpy array, but is not the best
format for handling this type of data which is very big. It does mean the matrices
are readable in their original format, but this is deprecated in favour of loading
direct from QUANT C# matrix dumps (binary) and saving as pickle of numpy matrix.
@param N This is the order of the matrix e.g. 7201 for the original QUANT 1 data
or 8436 for the QUANT 2 data containing England, Wales and Scotland
"""
#def acquireQ1CostMatricesFromCSV(N):
#    #download csv files from links and convert to pickle format for Python
#
#    #road
#    print('databuilder.py: downloading from ',url_QUANT1_CijRoadMin_csv)
#    tmpPath = os.path.join(tmpDataDir,'dijRoadMin.csv')
#    urllib.request.urlretrieve(url_QUANT1_CijRoadMin_csv, tmpPath)
#    #now load it - note this is a csv matrix dump NOT csv training data
#    with open(tmpPath) as file:
#        disRoadMin = np.zeros(N*N).reshape(N, N)
#        disRoadMin = np.loadtxt(file, delimiter=",")
#        filename = os.path.join(modelRunsDir,CijRoadMinFilename)
#        print('databuilder.py: saving file ',filename)
#        saveMatrix(disRoadMin,filename)
#
#    #bus
#    print('databuilder.py: downloading from ',url_QUANT1_CijBusMin_csv)
#    tmpPath = os.path.join(tmpDataDir,'dijBusMin.csv')
#    urllib.request.urlretrieve(url_QUANT1_CijBusMin_csv, tmpPath)
#    #now load it - note this is a csv matrix dump NOT csv training data
#    with open(tmpPath) as file:
#        disBusMin = np.zeros(N*N).reshape(N, N)
#        disBusMin = np.loadtxt(file, delimiter=",")
#        filename = os.path.join(modelRunsDir,CijBusMinFilename)
#        print('databuilder.py: saving file ',filename)
#        saveMatrix(disBusMin,filename)
#
#    #rail
#    print('databuilder.py: downloading from ',url_QUANT1_CijRailMin_csv)
#    tmpPath = os.path.join(tmpDataDir,'dijRailMin.csv')
#    urllib.request.urlretrieve(url_QUANT1_CijRailMin_csv, tmpPath)
#    #now load it - note this is a csv matrix dump NOT csv training data
#    with open(tmpPath) as file:
#        disRailMin = np.zeros(N*N).reshape(N, N)
#        disRailMin = np.loadtxt(file, delimiter=",")
#        filename = os.path.join(modelRunsDir,CijRailMinFilename)
#        print('databuilder.py: saving file ',filename)
#        saveMatrix(disRailMin,filename)
###def acquireQ1CostMatricesFromCSV

"""
Load data from the raw QUANT C Sharp binary format, which is just a dump
of M and N dimensions as 4 byte IEEE754 floats, followed by rows and columns
of data as 4 byte floats. The utils library functions gets the row ordering
right.
We test whether the files exist first in modelRuns dir, and only create if
they are missing.
NOTE: we keep the C Sharp binary files in the model runs dir as they can be
uesful, but the csv version of these in the other function are put into the
temporary download location and are deleted as they're not very useful.
@param dataDir
"""
def acquireQ1CostMatricesFromQCSBinary(dataDir):
    #road
    ensureMatrixFileQUANTtoPickle(
        PyCijRoadMinFilename,dataDir,url_QUANT1_CijRoadMin_qcs,QUANTCijRoadMinFilename)
    #bus
    ensureMatrixFileQUANTtoPickle(
        PyCijBusMinFilename,dataDir,url_QUANT1_CijBusMin_qcs,QUANTCijBusMinFilename)
    #rail
    ensureMatrixFileQUANTtoPickle(
        PyCijGBRailMinFilename,dataDir,url_QUANT1_CijGBRailMin_qcs,QUANTCijGBRailMinFilename)
    #CrowflyKM distances too? It's not really compatible with the other files
    #as it's in KM where they are in minutes
###acquireQ1CostMatricesFromQCSBinary


################################################################################

"""
Download and install QUANT1 training data for the GISTAM 2019 paper from
sets of csv training data for each mode type. This downloads the data from
the Open Science Foundation QUANT1 archive as training data csv sets for road
bus and rail. These contain the flows and the cost functions for the flows.
The csv training data is then broken down into Py_TObs_road.bin and
Py_Cij_road.bin (etc bus/rail) matrices in the data/fromQUANT directory.
These matrices are pickle dumps which load fast in Python. This was the
original way of getting the data from QUANT1 into python, but later on
the individual matrices were used and loaded direct from the QUANT C Sharp
format into Python. The training data dumps were produced direct from
the QUANT1 code and make a lot of sense for AI training sets. The binary
matrices make inter-operability with the main CS version of QUANT easier
though. The additional "quant1Install" function does the same as this the
hard way, by downloading and converting individual matrix binaries and
building people flows direct from the UK Census csv file.
TODO: need to end up with the same data as if you had got it from a
training file e.g. Py_files in data/fromQUANT
@param dataDir the the directory where the 'live' data is being installed e.g. model-runs
Temporary data always goes to tmpDataDir global
"""
def quant1InstallFromTrainingData(dataDir):
    #make a temporary data directory if it doesn't already exist
    if not os.path.exists(tmpDataDir):
        os.mkdir(tmpDataDir)
    #and make a 'fromQUANT' dir inside that
    fromQUANTDir = os.path.join(tmpDataDir,'fromQUANT')
    if not os.path.exists(fromQUANTDir):
        os.mkdir(fromQUANTDir)

    #make sure we have the zone codes text file to match the matrix index numbers up to area codes
    ensurePlainFile(ZoneCodesFilename,modelRunsDir,url_QUANT1_ZoneCodesText)

    #ensure training data files present for road, bus, rail - download if missing
    ensurePlainZIPFile(TrainingDataRoadCSVFilename,fromQUANTDir,url_QUANT1_trainingdata_Road_csv)
    ensurePlainZIPFile(TrainingDataBusCSVFilename,fromQUANTDir,url_QUANT1_trainingdata_Bus_csv)
    ensurePlainZIPFile(TrainingDataGBRailCSVFilename,fromQUANTDir,url_QUANT1_trainingdata_GBRail_csv)

    #road NOTE if the TObs exists, but the Cij doesn't then we miss it - assumes all or nothing
    if os.path.isfile(os.path.join(modelRunsDir,PyTObsRoadFilename)): #could check the Cij too?
        print('databuilder.py: ',PyTObsRoadFilename,' exists, skipping')
    else:
        print('databuilder.py: creating ',PyTObsRoadFilename,' and ',PyCijRoadMinFilename)
        Cij, Tij = loadQUANTCSV(os.path.join(fromQUANTDir,TrainingDataRoadCSVFilename),7201)
        saveMatrix(Cij,os.path.join(modelRunsDir,PyCijRoadMinFilename))
        saveMatrix(Tij,os.path.join(modelRunsDir,PyTObsRoadFilename))

    #bus NOTE if the TObs exists, but the Cij doesn't then we miss it - assumes all or nothing
    if os.path.isfile(os.path.join(modelRunsDir,PyTObsBusFilename)): #could check the Cij too?
        print('databuilder.py: ',PyTObsBusFilename,' exists, skipping')
    else:
        print('databuilder.py: creating ',PyTObsBusFilename,' and ',PyCijBusMinFilename)
        Cij, Tij = loadQUANTCSV(os.path.join(fromQUANTDir,TrainingDataBusCSVFilename),7201)
        saveMatrix(Cij,os.path.join(modelRunsDir,PyCijBusMinFilename))
        saveMatrix(Tij,os.path.join(modelRunsDir,PyTObsBusFilename))

    #gbrail NOTE if the TObs exists, but the Cij doesn't then we miss it - assumes all or nothing
    if os.path.isfile(os.path.join(modelRunsDir,PyTObsGBRailFilename)): #could check the Cij too?
        print('databuilder.py: ',PyTObsGBRailFilename,' exists, skipping')
    else:
        print('databuilder.py: creating ',PyTObsGBRailFilename,' and ',PyCijGBRailMinFilename)
        Cij, Tij = loadQUANTCSV(os.path.join(fromQUANTDir,TrainingDataGBRailCSVFilename),7201)
        saveMatrix(Cij,os.path.join(modelRunsDir,PyCijGBRailMinFilename))
        saveMatrix(Tij,os.path.join(modelRunsDir,PyTObsGBRailFilename))

    #all done
###


################################################################################

"""
Download and install all matrices and data if using QUANT version 1 to match
the GISTAM 2019 paper, Accelerating Urban Models with AI.
This is the same data as "quant1InstallFromTrainingData", but it downloads and
converts individual files so you can change them if you want. It also creates
TObs matrices directly from the Census flow csv file so you can see how it is
done.
This is the experimental version of 'installFromTrainingData' for people who
want to make changes to the flow and cost matrices.
@param dataDir the the directory where the 'live' data is being installed e.g. model-runs
Temporary data always goes to tmpDataDir global
"""
def quant1FullInstall(dataDir):
    #make a temporary data directory if it doesn't already exist
    if not os.path.exists(tmpDataDir):
        os.mkdir(tmpDataDir)

    #and then download the origin destination table into it
    #Checks for existence of "TravelToWorkFilename" and downloads and unpacks it if not already present
    path = os.path.join(tmpDataDir,'wu03ew_msoa.csv')
    ensurePlainZIPFile(TravelToWorkFilename,tmpDataDir,urlWU03EU)

    #make sure we have the zone codes text file to match the matrix index numbers up to area codes
    ensurePlainFile(ZoneCodesFilename,dataDir,url_QUANT1_ZoneCodesText)

    #and load the areas from the zonecodes lookup file
    ZoneLookup = ZoneCodes.loadZoneLookup(os.path.join(dataDir,ZoneCodesFilename))

    #OK, now we have the data, make it into a suitable matrix

    #This is the data format
    #"Area of residence","Area of workplace","All categories: Method of travel to work","Work mainly at or from home","Underground, metro, light rail, tram","Train","Bus, minibus or coach","Taxi","Motorcycle, scooter or moped","Driving a car or van","Passenger in a car or van","Bicycle","On foot","Other method of travel to work"
    #E02000001,E02000001,1506,0,73,41,32,9,1,8,1,33,1304,4
    #E02000001,E02000014,2,0,2,0,0,0,0,0,0,0,0,0
    #E02000001,E02000016,3,0,1,0,2,0,0,0,0,0,0,0

    """
    This builds the TObs[1|2].bin files for mode 1 and mode 2.
    Pre: needs the travel to work csv file the ZoneCodes.bin lookup file in the model runs directory.
    Post: generates model-runs\TObs.bin, TObs_1.bin and TObs_2.bin
    @param name="TObsFilename" Either EWS_TObs.bin or TObs.bin
    @param name="TObs1Filename" TObs filename for mode 1 i.e. EWS_TObs_1.bin or TObs_1.bin
    @param name="TObs2Filename">TObs filename for mode 2 i.e. EWS_TObs_2.bin or TObs_2.bin
    """

    #def generateTripsMatrix(string ZoneCodesLookupFilename, string TObsFilename, string TObs1Filename, string TObs2Filename, string TObs3Filename):
    #"Area of residence", "Area of workplace", "All categories: Method of travel to work", "Work mainly at or from home", "Underground, metro, light rail, tram", "Train", "Bus, minibus or coach", "Taxi", "Motorcycle, scooter or moped", "Driving a car or van", "Passenger in a car or van", "Bicycle", "On foot", "Other method of travel to work"
    # first the all modes trip matrix
    if os.path.isfile(os.path.join(dataDir, PyTObsAllFilename)):
        print('databuilder.py:',PyTObsAllFilename,' exists - skipping')
    else:
        print('databuilder.py::Generating ',PyTObsAllFilename,'in ',dataDir)
        ModeAll = [ "All categories: Method of travel to work" ]
        TijObs = generateTripsMatrix(os.path.join(tmpDataDir, TravelToWorkFilename), ZoneLookup, ModeAll)
        saveMatrix(TijObs,os.path.join(dataDir, PyTObsAllFilename))

    #If you want two mode (private/public) data then this is it - uncomment this and comment
    #3 mode code following. NOTE: the filenames will need changing as they don't exist in globals
    #Two mode hasn't been used in any of the AI models so far.
    #now the split two modes trips matrices
    #Mode2_1 = [ "Taxi", "Motorcycle, scooter or moped", "Driving a car or van", "Passenger in a car or van", "Bicycle" ]
    #Mode2_2 = [ "Underground, metro, light rail, tram", "Train", "Bus, minibus or coach" ]
    ##NOTE: the two modes don't add up to the total as there are some cols missing - Work from home, on foot, other
    #if os.path.isfile(os.path.join(modelRunsDir,TObs21Filename)):
    #    print("databuilder.py:",TObs21Filename," exists - skipping")
    #else:
    #    print("databuilder.py::Generating ",TObs21Filename)
    #    TijObs1 = generateTripsMatrix(os.path.join(modelRunsDir,TravelToWorkFilename), ZoneLookup, Mode2_1)
    #    saveMatrix(TijObs1,os.path.join(modelRunsDir,TObs21Filename)) #todo: should be Py_
    #if os.path.isfile(os.path.join(modelRunsDir,TObs22Filename)):
    #    print("databuilder.py:",TObs22Filename," exists - skipping")
    #else:
    #    print("databuilder.py::Generating ",TObs22Filename)
    #    TijObs2 = generateTripsMatrix(os.path.join(modelRunsDir,TravelToWorkFilename), ZoneLookup, Mode2_2)
    #    saveMatrix(TijObs2,os.path.join(modelRunsDir,TObs22Filename)) #todo: should be Py_

    #three mode trip matrices (Road, Bus, Rail)
    Mode3_1 = [ "Taxi", "Motorcycle, scooter or moped", "Driving a car or van", "Passenger in a car or van" ]
    Mode3_2 = [ "Bus, minibus or coach" ]
    Mode3_3 = [ "Underground, metro, light rail, tram", "Train" ]
    if os.path.isfile(os.path.join(dataDir, PyTObsRoadFilename)):
        print('databuilder.py:',PyTObsRoadFilename,' exists - skipping')
    else:
        print('databuilder.py::Generating ',PyTObsRoadFilename,' in',dataDir)
        TijObs31 = generateTripsMatrix(os.path.join(tmpDataDir, TravelToWorkFilename), ZoneLookup, Mode3_1)
        saveMatrix(TijObs31,os.path.join(dataDir, PyTObsRoadFilename))
    if os.path.isfile(os.path.join(dataDir, PyTObsBusFilename)):
        print('databuilder.py:',PyTObsBusFilename,' exists - skipping')
    else:
        print("databuilder.py::Generating ",PyTObsBusFilename,' in',dataDir)
        TijObs32 = generateTripsMatrix(os.path.join(tmpDataDir, TravelToWorkFilename), ZoneLookup, Mode3_2)
        saveMatrix(TijObs32,os.path.join(dataDir, PyTObsBusFilename))
    if os.path.isfile(os.path.join(dataDir, PyTObsGBRailFilename)):
        print('databuilder.py:',PyTObsGBRailFilename,' exists - skipping')
    else:
        print("databuilder.py::Generating ",PyTObsGBRailFilename,' in',dataDir)
        TijObs33 = generateTripsMatrix(os.path.join(tmpDataDir, TravelToWorkFilename), ZoneLookup, Mode3_3)
        saveMatrix(TijObs33,os.path.join(dataDir, PyTObsGBRailFilename))

    ################################################################################
    # now cost matrices - all we can do here is use existing data from QUANT, but
    # convert to python
    # DEPRECATED - Either download zipped csv (BIG) matrices and convert to python
    # OR download the C Sharp binaries and convert them to pickle directly - this
    # is easier and is the preferred method.
    ################################################################################

    #now download and convert the QUANT cost matrices into python ones
    #acquireQ1CostMatricesFromCSV(7201) #I'm considering this deprecated, but it should work (N=7201 or 8436? too)
    acquireQ1CostMatricesFromQCSBinary(dataDir) #acquire from QUANT CS binaries
    #end of QUANT 1 cost matrix conversions

    #I think that's everything...
###

################################################################################

"""
Download and install all the data for the updated QUANT version 2, which
also contains Scotland data.
Rather than building the flow matrices from the Census file, just download
the matrices from the Open Science Foundation data dump. This makes a lot
more sense as the QUANT 2 data contains Scotland flows which are in a different
file to the England and Wales one. Using the pre-computed matrices like this
means that the real QUANT2 can do the hard part of stitching all the data back
together again. If you want to know where the data comes from, the flows including
Scotland are in the file you can download from wicid directly:
https://wicid.ukdataservice.ac.uk
TODO: convert this to using the training data and just mention that the individual
files are in the archive too.
"""
def quant2Install():
    #make a temporary data directory if it doesn't already exist
    if not os.path.exists(tmpDataDir):
        os.mkdir(tmpDataDir)

    #first off the TObs trip flow matrices
    #mode 1 = road, mode 2 = bus, mode 3 = rail

    #NOTE: the Tij matrices always get downloaded as csharp binary dumps, converted
    #to numpy matrices, pickled and saved over the top of the original downloads

    #Tij road, mode 1
    ensureMatrixFileQUANTtoPickle(TObsRoadFilename, tmpDataDir, url_QUANT2_TijRoad, TObsRoadFilename)
    #Tij bus
    ensureMatrixFileQUANTtoPickle(TObsBusFilename, tmpDataDir, url_QUANT2_TijBus, TObsBusFilename)
    #Tij rail
    ensureMatrixFileQUANTtoPickle(TObsRailFilename, tmpDataDir, url_QUANT2_TijRail, TObsRailFilename)

    #now the cost files

    #Cij road
    ensureMatrixFileQUANTtoPickle(CijRoadFilename, tmpDataDir, url_QUANT2_CijRoadMin, QUANTCijRoadMinFilename)
    #Cij bus
    ensureMatrixFileQUANTtoPickle(CijBusFilename, tmpDataDir, url_QUANT2_CijBusMin, QUANTCijBusMinFilename)
    #Cij rail
    ensureMatrixFileQUANTtoPickle(CijRailFilename, tmpDataDir, url_QUANT2_CijRailMin, QUANTCijRailMinFilename)

    #and finally the zone codes lookup

    #zonecodes text file
    ensurePlainFile(ZoneCodesFilename, tmpDataDir, url_QUANT2_ZoneCodesText)

    #and that's all there is to it...
###

"""
Perform a clean of the matrices in the modelruns directory. This ONLY removes the files
specified in the list below, which are the trips, costs, zonecodes and raw travel to work
files.
"""
def clean():
    #todo: FIX!
    print('databuilder: clean')
    files = [
        TravelToWorkFilename,
        ZoneCodesFilename,
        PyCijRoadMinFilename, PyCijBusMinFilename, PyCijGBRailMinFilename,
        PyTObsAllFilename,
        PyTObsRoadFilename, PyTObsBusFilename, PyTObsGBRailFilename,
        TObsFilename,
        TObs21Filename, TObs22Filename,
        TObs31Filename, TObs32Filename, TObs33Filename,
        QUANTCijRoadMinFilename, QUANTCijBusMinFilename, QUANTCijRailMinFilename
    ]
    fromQUANTDir = os.path.join(tmpDataDir,'fromQUANT')
    for file in files:
        path = os.path.join(tmpDataDir, file)
        if os.path.isfile(path):
            print('databuilder: deleting ',path) 
            #os.remove(path) todo: check it first!!!
        path = os.path.join(fromQUANTDir,file)
        if os.path.isfile(path):
            print('databuilder: deleting ',path) 
            #os.remove(path) todo: check it first!!!
###

################################################################################
# Main Program
# Either install QUANT1 or QUANT2, depending on what data you want.
# You probably want QUANT2 with the latest data in it unless you're trying
# to replicate the GISTAM 2019 paper exactly.
################################################################################

#NOTE: QUANT1 and QUANT2 matrices will overwrite each other, so can't coexist together
#This means that you MUST CLEAN the model runs dir of the matrices if you swap from
#one version to the other. In practice though, Q1 is deprecated in favour of Q2.

#clean() #if you want to remove all the files and start again from scratch e.g. change from QUANT 1 to 2

#make sure we have a model runs dir
if not os.path.exists(modelRunsDir):
    os.mkdir(modelRunsDir)

#install either QUANT1 or QUANT2 files - probably QUANT2

################################################################################
# QUANT 1 GISTAM 2019
################################################################################

#You want the training data install (first one) if you're looking for GISTAM2019 data
#quant1InstallFromTrainingData(modelRunsDir) #matches GISTAM2019 paper - installs from three pre-built zip archives
#EXPERIMENTAL
quant1FullInstall(modelRunsDir) #same data as above, but goes through the process of creation i.e. you could change it

################################################################################
# QUANT 2 New Work
################################################################################

#quant2Install() #new work - you almost certainly want this one