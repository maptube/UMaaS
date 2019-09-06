#test program

import random
import time
import networkx as nx

from graphserver.PublicTransportNetwork import PublicTransportNetwork
from zonecodes import ZoneCodes

from graphserver.DijkstraSSSP import DijkstraSSSP

EnglandWalesMSOA_WGS84 = 'data/geometry/EnglandWalesScotland_MSOAWGS84.shp'

################################################################################

"""
This is a test of the one time build of the rail network data
"""
def graphtest1():
    print("graphserver/graphtest.py graphtest1: Test of one time build for rail network data using GPU")

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

def graphtest2():
    print("graphserver/graphtest.py graphtest2: Test of one time build for bus network data using GPU")

    print("initialise ptn")
    ptn = PublicTransportNetwork()
    ptn.initialiseGTFSNetworkGraph([
        'data/bus/gtfs/EA',
        'data/bus/gtfs/EM',
        'data/bus/gtfs/L',
        'data/bus/gtfs/NCSD',
        'data/bus/gtfs/NE',
        'data/bus/gtfs/NW',
        'data/bus/gtfs/S',
        'data/bus/gtfs/SE',
        'data/bus/gtfs/SW',
        'data/bus/gtfs/W',
        'data/bus/gtfs/WM',
        'data/bus/gtfs/Y'
        ],
        [True, True, False, True, True, True, False, False]
    ) #Tram = 0, Subway = 1, Rail = 2, Bus = 3, Ferry = 4, CableCar = 5, Gondola = 6, Funicular = 7
    count = ptn.FixupWalkingNodes(500)
    print("GenerateBusMatrix, after fixup walking nodes: ", str(ptn.graph.number_of_nodes())," vertices and ", str(ptn.graph.number_of_edges()), " edges in network.")
    ptn.writeGraphML("model-runs/graph_bus_EWS.graphml")
    ptn.writeVertexPositions("model-runs/vertices_bus_EWS.csv") #note: you also need the vertex position map, otherwise you don't know where the graphml vertices are

    print("zone codes")
    #ZoneLookup is the mapping between the zone code numbers and the MSOA code and lat/lon
    ZoneLookup = ZoneCodes.fromFile() #NOTE: this is the class, you probably want to use it as ZoneLookup.dt?

    print("bus centroids")
    #BusCentroidLookup is a mapping (Dict<string,string>) between the zone's MSOA code and the graph node to use for this zone
    BusCentroidLookup = ptn.FindCentroids(ZoneLookup.dt, EnglandWalesMSOA_WGS84, 'MSOA11CD') #NOTE WGS84 shapefile!
    ptn.saveCentroids(BusCentroidLookup,"model-runs/centroids_bus_EWS.csv")
    #add walking centroids

    shortestPaths = ptn.TestCUDASSSP(ZoneLookup.dt,BusCentroidLookup)
    #for k,v in shortestPaths.items():
    #    print("SSSP",k,v)

    #and this is the result...
    #Running SSSP: n=332720, nnz=1980701
    #SSSP Elapsed milliseconds = 1009.234375 1013.9092801965704 validcount= 8432
    #
    #1000 milliseconds for 1 equates to 7201 seconds for all, or about 2 hours

    #this is a baseline check against how fast the network x implementation of Dijkstra could do the same thing
    #start = time.clock()
    #result = nx.all_pairs_dijkstra_path_length(ptn.graph,weight='weight')
    #secsAPSP = time.clock()-start
    #print("graphtest2: NetworkX all_pairs_dijkstra_path_length test: ",secsAPSP," seconds")

################################################################################

def graphtestSSSP():
    #this is a shortest path SSSP test using dijkstra
    #make a test network that's not very big...
    graph = nx.MultiDiGraph() # note weighted directed
    graph.add_edge("A","B",weight=1)
    graph.add_edge("A","E",weight=3)
    graph.add_edge("B","C",weight=2)
    graph.add_edge("B","E",weight=2)
    graph.add_edge("C","D",weight=1)
    graph.add_edge("C","E",weight=1)
    graph.add_edge("D","E",weight=1)
    graph.add_edge("D","F",weight=2)
    graph.add_edge("F","E",weight=1)
    #cycle
    graph.add_edge("E","A",weight=1)
    sssp = DijkstraSSSP(graph)
    sssp.dijkstraSSSPTest("A")
    sssp.debugPrintData()

    print("\n")

    #and now with the APSP KKP algorithm
    print("KKP_APSP test")
    sssp.KKP_APSP()
    sssp.debugPrintAPSPData()

################################################################################

def randomGraph(v,e):
    #generate a random graph with the required number of vertices and edges
    #NOTE: if e<v, then you will still get one edge for every vertex i.e. e=v
    #This is by design to ensure that all the nodes are actually connected.
    graph = nx.MultiDiGraph()
    #if we want to guarantee that every node is connected to something, then the best
    #way is to start by creating an edge from each individual node to a random node
    numEdges = 0
    for i in range(0,v):
        src = "N"+str(i)
        j = random.randint(0,v-1)
        dest = "N"+str(j)
        weight = random.uniform(1,10) #uniform weights between 1 and 10
        graph.add_edge(src,dest,weight=weight)
        numEdges+=1
    #endif

    #OK, now fill in the rest of the required edges completely randomly
    while numEdges<e:
        i = random.randint(0,v-1)
        src = "N"+str(i)
        j = random.randint(0,v-1)
        dest = "N"+str(j)
        weight = random.uniform(1,10) #uniform weights between 1 and 10
        graph.add_edge(src,dest,weight=weight)
        numEdges+=1
    #endwhile

    return graph

################################################################################

def graphtestRandomGraphs():
    #this is a test of a large random graph, computed using SSSP Dijkstra against KKP APSP modified Dijkstra
    #the aim is to quantify the benefit of running APSP against multiple SSSP runs with different origins
    
    numVertices = 3165 #3165
    numEdges=10269 #10269

    print("graphTestRandomGraphs: SSSP APSP comparison test n=",numVertices," e=",numEdges)

    graph = randomGraph(numVertices,numEdges)

    sssp = DijkstraSSSP(graph)
    start = time.process_time()
    #runtimes = 100 #this is the number of repeat times I run the SSSP to get an average
    for i in range(0,numVertices):
        sssp.dijkstraSSSPTest("N"+str(i))
    secsSSSP = time.process_time()-start
    #secsSSSP=secsSSSP/runtimes #note dividing the time by the number of runs above - get this right 
    print("graphTestRandomGraphs: SSSP test ",secsSSSP," secs")

    #now the APSP comparison

    start = time.process_time()
    #for i in range(0,100):
    sssp.KKP_APSP()
    secsAPSP = time.process_time()-start
    print("graphTestRandomGraphs: KKP_APSP test ",secsAPSP," secs")
    print("graphTestRandomGraphs: ratio ",secsAPSP/secsSSSP)
    

################################################################################

def graphtestNetworkX():
    #this is a test of the in-built networkx functions
    #see: https://networkx.github.io/documentation/stable/reference/algorithms/shortest_paths.html
    #IMPORTANT: some of the network x functions return generators to the data. This means that you MUST
    #iterate over all the data in order to get a representative timing value. In other words, the function
    #only prepares the shortest paths to run. The real code gets deferred until when you request the data.

    numVertices = 3165 #3165
    numEdges=10269 #10269

    print("graphTestNetworkX: APSP comparison test using in-built network x functions n=",numVertices," e=",numEdges)

    print("building random graph")
    graph = randomGraph(numVertices,numEdges)
    print("finished building random graph")

    start = time.clock()
    #for n in range(0,1000):
        #print(n)
    #result = nx.all_pairs_dijkstra_path_length(graph,weight='weight') #58s
    #result = nx.all_pairs_bellman_ford_path_length(graph,weight='weight') #200s
    result = nx.all_pairs_shortest_path_length(graph) #57.5s
    #result = nx.floyd_warshall(graph) #takes forever
    result2 = {}
    for key, data in result:
        result2[key] = data
        #print("key=",key,"data=",data)
    secsAPSP = time.clock()-start
    print("finish")
    print("graphTestNetworkX: all_pairs_dijkstra_path_length test ",secsAPSP," secs")

