"""
databuilder.py
Builds the data needed by the test models
"""

import os;
import os.path;
import zipfile;
import urllib.request;
import numpy as np;

#code to load goes here
#1. download wu03ew_v2.csv if it's not already here
#https://www.nomisweb.co.uk/output/census/2011/wu03ew_msoa.zip

tmpDataDir = 'data'
urlWU03EU = 'https://www.nomisweb.co.uk/output/census/2011/wu03ew_msoa.zip'

#make a data directory if it doesn't already exist
if not os.path.exists(tmpDataDir):
    os.mkdir(tmpDataDir)

#and then download the origin destination table into it
if not os.path.isfile(tmpDataDir+'/wu03ew_msoa.csv'):
    #download zip and unzip it
    print("downloading from: ",urlWU03EU)
    urllib.request.urlretrieve (urlWU03EU, tmpDataDir+'/wu03ew_msoa.zip')
    #unzip
    zip_ref = zipfile.ZipFile(tmpDataDir+'/wu03ew_msoa.zip', 'r')
    zip_ref.extractall(tmpDataDir)
    zip_ref.close()

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
def generateTripsMatrix(string ZoneCodesLookupFilename, string TObsFilename, string TObs1Filename, string TObs2Filename, string TObs3Filename):
    #this is the data root dir from app.config
    string ModelRunsDir = ConfigurationManager.AppSettings["ModelRunsDir"];
    
    DataTable ZoneLookup = (DataTable)Serialiser.Get(Path.Combine(ModelRunsDir,ZoneCodesLookupFilename));
    #"Area of residence", "Area of workplace", "All categories: Method of travel to work", "Work mainly at or from home", "Underground, metro, light rail, tram", "Train", "Bus, minibus or coach", "Taxi", "Motorcycle, scooter or moped", "Driving a car or van", "Passenger in a car or van", "Bicycle", "On foot", "Other method of travel to work"
    # first the all modes trip matrix
    string[] ModeAll = {
                                "All categories: Method of travel to work"
                                 };
            FMatrix TijObs = DataBuilder.GenerateTripsMatrix(Path.Combine(ModelRunsDir, TravelToWorkFilename), ZoneLookup, ModeAll);
            TijObs.DirtySerialise(Path.Combine(ModelRunsDir, TObsFilename));

            //now the split two modes trips matrices
            string[] Mode2_1 = {
                                  "Taxi",
                                  "Motorcycle, scooter or moped",
                                  "Driving a car or van",
                                  "Passenger in a car or van",
                                  "Bicycle"
                              };
            string[] Mode2_2 = {
                                 "Underground, metro, light rail, tram",
                                 "Train",
                                 "Bus, minibus or coach"
                              };
            //NOTE: the two modes don't add up to the total as there are some cols missing - Work from home, on foot, other
            FMatrix TijObs1 = DataBuilder.GenerateTripsMatrix(Path.Combine(ModelRunsDir,TravelToWorkFilename), ZoneLookup, Mode2_1);
            TijObs1.DirtySerialise(Path.Combine(ModelRunsDir,TObs1Filename));
            FMatrix TijObs2 = DataBuilder.GenerateTripsMatrix(Path.Combine(ModelRunsDir,TravelToWorkFilename), ZoneLookup, Mode2_2);
            TijObs2.DirtySerialise(Path.Combine(ModelRunsDir,TObs2Filename));

            //and while we're here, the three mode trip matrices as well (Road, Bus, Rail)
            string[] Mode3_1 = {
                                   "Taxi",
                                   "Motorcycle, scooter or moped",
                                   "Driving a car or van",
                                   "Passenger in a car or van"
                                };
            string[] Mode3_2 = {
                                   "Bus, minibus or coach"
                                };
            string[] Mode3_3 = {
                                   "Underground, metro, light rail, tram",
                                   "Train"
                               };
            FMatrix TijObs31 = DataBuilder.GenerateTripsMatrix(Path.Combine(ModelRunsDir, TravelToWorkFilename), ZoneLookup, Mode3_1);
            TijObs31.DirtySerialise(Path.Combine(ModelRunsDir, TObs1Filename));
            FMatrix TijObs32 = DataBuilder.GenerateTripsMatrix(Path.Combine(ModelRunsDir, TravelToWorkFilename), ZoneLookup, Mode3_2);
            TijObs32.DirtySerialise(Path.Combine(ModelRunsDir, TObs2Filename));
            FMatrix TijObs33 = DataBuilder.GenerateTripsMatrix(Path.Combine(ModelRunsDir, TravelToWorkFilename), ZoneLookup, Mode3_3);
            TijObs33.DirtySerialise(Path.Combine(ModelRunsDir, TObs3Filename));

        }
