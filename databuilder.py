"""
databuilder.py
Builds the data needed by the test models
Run this file first to auto generate all the trips matrices and raw data needed to run the models
"""

import os
import os.path
import zipfile
import urllib.request
import numpy as np
from shutil import copyfile
from globals import *
from utils import loadZoneLookup
from utils import generateTripsMatrix
from utils import loadMatrix, saveMatrix, loadQUANTMatrix

#code to load goes here
#1. download wu03ew_v2.csv if it's not already here
#https://www.nomisweb.co.uk/output/census/2011/wu03ew_msoa.zip

tmpDataDir = 'data'
urlWU03EU = 'https://www.nomisweb.co.uk/output/census/2011/wu03ew_msoa.zip'




#make a data directory if it doesn't already exist
if not os.path.exists(tmpDataDir):
    os.mkdir(tmpDataDir)

#and then download the origin destination table into it
if not os.path.isfile(os.path.join(tmpDataDir,'wu03ew_msoa.csv')):
    #download zip and unzip it
    print("downloading from: ",urlWU03EU)
    urllib.request.urlretrieve (urlWU03EU, os.path.join(tmpDataDir,'wu03ew_msoa.zip'))
    #unzip
    zip_ref = zipfile.ZipFile(os.path.join(tmpDataDir,'wu03ew_msoa.zip'), 'r')
    zip_ref.extractall(tmpDataDir)
    zip_ref.close()
    copyfile(os.path.join(tmpDataDir,ZoneCodesFilename),os.path.join(modelRunsDir,ZoneCodesFilename)) #copy the tmp data location file to model-runs where everything is
    #and the file it outputs will be called TravelToWorkFilename, defined above, but I can't control this - it's whatever's in the zip

#OK, now we have the data, make it into a suitable matrix

ZoneLookup = loadZoneLookup(os.path.join(modelRunsDir,ZoneCodesFilename))

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
if os.path.isfile(os.path.join(modelRunsDir, TObsFilename)):
    print("databuilder.py:",TObsFilename," exists - skipping")
else:
    print("databuilder.py::Generating ",TObsFilename)
    ModeAll = [ "All categories: Method of travel to work" ]
    TijObs = generateTripsMatrix(os.path.join(modelRunsDir, TravelToWorkFilename), ZoneLookup, ModeAll)
    saveMatrix(TijObs,os.path.join(modelRunsDir, TObsFilename))

#now the split two modes trips matrices
Mode2_1 = [ "Taxi", "Motorcycle, scooter or moped", "Driving a car or van", "Passenger in a car or van", "Bicycle" ]
Mode2_2 = [ "Underground, metro, light rail, tram", "Train", "Bus, minibus or coach" ]
#NOTE: the two modes don't add up to the total as there are some cols missing - Work from home, on foot, other
if os.path.isfile(os.path.join(modelRunsDir,TObs21Filename)):
    print("databuilder.py:",TObs21Filename," exists - skipping")
else:
    print("databuilder.py::Generating ",TObs21Filename)
    TijObs1 = generateTripsMatrix(os.path.join(modelRunsDir,TravelToWorkFilename), ZoneLookup, Mode2_1)
    saveMatrix(TijObs1,os.path.join(modelRunsDir,TObs21Filename))
if os.path.isfile(os.path.join(modelRunsDir,TObs22Filename)):
    print("databuilder.py:",TObs22Filename," exists - skipping")
else:
    print("databuilder.py::Generating ",TObs22Filename)
    TijObs2 = generateTripsMatrix(os.path.join(modelRunsDir,TravelToWorkFilename), ZoneLookup, Mode2_2)
    saveMatrix(TijObs2,os.path.join(modelRunsDir,TObs22Filename))

#and while we're here, the three mode trip matrices as well (Road, Bus, Rail)
Mode3_1 = [ "Taxi", "Motorcycle, scooter or moped", "Driving a car or van", "Passenger in a car or van" ]
Mode3_2 = [ "Bus, minibus or coach" ]
Mode3_3 = [ "Underground, metro, light rail, tram", "Train" ]
if os.path.isfile(os.path.join(modelRunsDir, TObs31Filename)):
    print("databuilder.py:",TObs31Filename," exists - skipping")
else:
    print("databuilder.py::Generating ",TObs31Filename)
    TijObs31 = generateTripsMatrix(os.path.join(modelRunsDir, TravelToWorkFilename), ZoneLookup, Mode3_1)
    saveMatrix(TijObs31,os.path.join(modelRunsDir, TObs31Filename))
if os.path.isfile(os.path.join(modelRunsDir, TObs32Filename)):
    print("databuilder.py:",TObs32Filename," exists - skipping")
else:
    print("databuilder.py::Generating ",TObs32Filename)
    TijObs32 = generateTripsMatrix(os.path.join(modelRunsDir, TravelToWorkFilename), ZoneLookup, Mode3_2)
    saveMatrix(TijObs32,os.path.join(modelRunsDir, TObs32Filename))
if os.path.isfile(os.path.join(modelRunsDir, TObs33Filename)):
    print("databuilder.py:",TObs33Filename," exists - skipping")
else:
    print("databuilder.py::Generating ",TObs33Filename)
    TijObs33 = generateTripsMatrix(os.path.join(modelRunsDir, TravelToWorkFilename), ZoneLookup, Mode3_3)
    saveMatrix(TijObs33,os.path.join(modelRunsDir, TObs33Filename))

#now cost matrices - all we can do here is use existing data from QUANT, but convert to python
#todo: download zipped csv (BIG) matrices and convert to python

#now convert the QUANT cost matrices into python ones

if os.path.isfile(os.path.join(modelRunsDir, CijRoadMinFilename)):
    print("databuilder.py:",CijRoadMinFilename," exists - skipping")
else:
    print("databuilder.py::Generating ",CijRoadMinFilename)
    CijRoad = loadQUANTMatrix(os.path.join(modelRunsDir, QUANTCijRoadMinFilename))
    saveMatrix(CijRoad,os.path.join(modelRunsDir,CijRoadMinFilename))

if os.path.isfile(os.path.join(modelRunsDir, CijBusMinFilename)):
    print("databuilder.py:",CijBusMinFilename," exists - skipping")
else:
    print("databuilder.py::Generating ",CijBusMinFilename)
    CijBus = loadQUANTMatrix(os.path.join(modelRunsDir, QUANTCijBusMinFilename))
    saveMatrix(CijBus,os.path.join(modelRunsDir,CijBusMinFilename))

if os.path.isfile(os.path.join(modelRunsDir, CijRailMinFilename)):
    print("databuilder.py:",CijRailMinFilename," exists - skipping")
else:
    print("databuilder.py::Generating ",CijRailMinFilename)
    CijRail = loadQUANTMatrix(os.path.join(modelRunsDir, QUANTCijRailMinFilename))
    saveMatrix(CijRail,os.path.join(modelRunsDir,CijRailMinFilename))

#end of QUANT cost matrix conversions

#I think that's everything...