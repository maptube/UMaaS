#test program

from PublicTransportNetwork import PublicTransportNetwork

EnglandWalesMSOA_WGS84 = 'data/geometry/EnglandWalesScotland_MSOAWGS84.shp'

#todo: need the zone lookup here

ptn = PublicTransportNetwork()
#ptn.initialiseGTFSNetworkGraph(['data/rail'], [False, True, True, False, True, False, False, False]) #Tram = 0, Subway = 1, Rail = 2, Bus = 3, Ferry = 4, CableCar = 5, Gondola = 6, Funicular = 7
#count = ptn.FixupWalkingNodes(500)
ZoneLookup = {} #HACK!
RailCentroidLookup = ptn.FindCentroids(ZoneLookup, EnglandWalesMSOA_WGS84, 'MSOA11CD') #NOTE WGS84 shapefile!
#add walking centroids