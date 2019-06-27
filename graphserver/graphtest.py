#test program

from graphserver.PublicTransportNetwork import PublicTransportNetwork
from zonecodes import ZoneCodes

EnglandWalesMSOA_WGS84 = 'data/geometry/EnglandWalesScotland_MSOAWGS84.shp'

################################################################################

"""
This is a test of the one time build of the rail network data
"""
def graphtest1():
    print("graphserver/graphtest1.py graphtest1: Test of one time build for rail network data using GPU")
    #todo: need the zone lookup here

    print("initialise ptn")
    ptn = PublicTransportNetwork()
    ptn.initialiseGTFSNetworkGraph(['data/rail'], [False, True, True, False, True, False, False, False]) #Tram = 0, Subway = 1, Rail = 2, Bus = 3, Ferry = 4, CableCar = 5, Gondola = 6, Funicular = 7
    count = ptn.FixupWalkingNodes(500)

    print("zone codes")
    #ZoneLookup is the mapping between the zone code numbers and the MSOA code and lat/lon
    ZoneLookup = ZoneCodes.fromFile() #NOTE: this is the class, you probably want to use it as ZoneLookup.dt?

    print("rail centroids")
    #RailCentroidLookup is a mapping (Dict<string,string>) between the zone's MSOA code and the graph node to use for this zone
    RailCentroidLookup = ptn.FindCentroids(ZoneLookup.dt, EnglandWalesMSOA_WGS84, 'MSOA11CD') #NOTE WGS84 shapefile!
    #add walking centroids

    shortestPaths = ptn.TestCUDASSSP(ZoneLookup.dt,RailCentroidLookup)
    #for k,v in shortestPaths.items():
    #    print("SSSP",k,v)

################################################################################