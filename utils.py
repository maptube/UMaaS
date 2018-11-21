"""
utils.py
Data building utilities
"""

import numpy as np;
import csv;


def loadZoneLookup(filename):
    ZoneLookup = {}
    with open(filename) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        header = next(reader,None)
        for row in reader:
            #zonei,areakey,name,east,north
            #0,E02000001,City of London 001,532482.7,181269.3
            zonei = int(row[0])
            msoa = row[1]
            name = row[2]
            east = float(row[4])
            north = float(row[5])
            





"""
Build a trips matrix from data in the Census CSV file (i.e. wu03ew_v2.csv).
The point behind this is to be able to build separate matrices from the sums of a user defined set of columns.
This does assume that the two area keys for the origin destination areas are the first two columns in the file.
NOTE: this defines which way round the i and j are. See comment in code main loop below.
@param name="CSVFilename"
@param name="ZoneLookup" Lookup between MSOA area key and ZoneCode index number
@param name="ColumnNames" Names of columns added together to get the total trips i.e. "UNDERGROUND", "TRAIN", "BUS". These must be in the CSV header line
@returns An observed trips matrix
"""
def generateTripsMatrix(string CSVFilename, DataTable ZoneLookup, string [] ColumnNames):
    N = ZoneLookup.Rows.Count
    TijObs = np.matrix(N,N)

    with open(CSVFilename) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        Fields = next(reader,None)
        #work out column index numbers for all the column names
        ColI = []
        for i = in Range(ColumnNames.Length):
             ColI[i] = -1
        for i in Range(Fields.Length):
            Field = Fields[i]
            for j in Range(ColumnNames.Length):
                if (Field == ColumnNames[j]):
                    ColI[j] = i

        #OK, now on to the data
        #"Area of residence","Area of workplace","All categories: Method of travel to work","Work mainly at or from home","Underground, metro, light rail, tram","Train","Bus, minibus or coach","Taxi","Motorcycle, scooter or moped","Driving a car or van","Passenger in a car or van","Bicycle","On foot","Other method of travel to work"
        #E02000001,E02000001,1506,0,73,41,32,9,1,8,1,33,1304,4
        LineCount = 1
        for row in reader:
        {
            LineCount+=1
            #Fields = Line.Split(new char[] { ',' });
            ZoneR = row[0]
            ZoneW = row[1]
            Sum = 0
            for i = in Range(ColI.Length):
                Count = int(row[ColI[i]])
                Sum += Count;
            DataRow RowR = ZoneLookup.Rows.Find(ZoneR); //could potentially fail if ZoneR or ZoneW didn't exist in the shapefile
            DataRow RowW = ZoneLookup.Rows.Find(ZoneW);
            if ((RowR == null) || (RowW == null))
            {
                print("Warning: trip " + ZoneR + " to " + ZoneW + " zones not found - skipped")
                continue
            }
            int ZoneR_idx = (int)RowR["zonei"];
            int ZoneW_idx = (int)RowW["zonei"];
            #TijObs._M[ZoneR_idx, ZoneW_idx] = Sum; #this was the original that was apparently the wrong way around
            TijObs._M[ZoneW_idx, ZoneR_idx] = Sum; #this is the above line with i and j flipped - right way around
        }
        print("Loaded " + CSVFilename + " and processed " + LineCount + " lines of data")
        print("Finished GenerateTripsMatrix")
    }

    return TijObs

#######################################################################################################################################