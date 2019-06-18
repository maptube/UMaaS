"""
PublicTransportNetwork
Wrapper for a network built from GTFS data
"""

import sys
import networkx as nx
from scipy.spatial import KDTree, cKDTree
#from sklearn.neighbors import KDTree
#from sklearn.neighbors import KNeighborsClassifier
#geopandas?
import geopandas as gpd
#from rtree import index
import math

from GTFSUtils import GTFSUtils

class PublicTransportNetwork:

################################################################################
# gtfs properties
    #public AdjacencyGraph<string, WeightedEdge<string>> graph; //using QuickGraph structures
    #public Dictionary<Edge<string>, double> costs; //this is the distance weight list for edges

    #public KdTree<string> kdtree; //this is for finding the closest graph vertex to each msoa centroid

    
    # Return the number of vertices in the graph.
    #public int VertexCount {
    #    get { return graph.VertexCount;  }
    #}

    
    # Return the number of edges in the graph.
    #public int EdgeCount
    #{
    #    get { return graph.EdgeCount; }
    #}

    def __init__(self):
        self.graph = nx.Graph()
        self.vertices = {}
        self.vidx = {} #map between vertex number (to match kdtree) and vertex string (to match graph)
        #self.costs
        self.kdtree = None

################################################################################


    """
    Build the graph of the bus network which we can then use to run queries on.
    @param name="GTFSDirectories" (string [])
    @param name="AllowedRouteTypes" (boolean []) Bitset of RouteTypes which will be included in the following order: Tram=0, Subway=1, Rail=2, Bus=3, Ferry=4, CableCar=5, Gondola=6, Funicular=7</param>
    """
    def initialiseGTFSNetworkGraph(self, GTFSDirectories, AllowedRouteTypes):
        #graph = new AdjacencyGraph<string, WeightedEdge<string>>(); //using QuickGraph structures
        self.graph = nx.MultiDiGraph() #we need directed and parallel edges
        #costs = new Dictionary<Edge<string>, double>(); //this is the distance weight list for edges

        #kdtree = new KdTree<string>(); //this is for finding the closest graph vertex to each msoa centroid
        kdtreedata = [] #because we have to make the kdtree index in the constructor, we need to generate a list of [node_id,lat,lon] for every vertex
        #self.kdtree = index.Index() #rtree index

        for dir in GTFSDirectories:
            #Dictionary<string, List<GTFS_ODLink>> links = new Dictionary<string, List<GTFS_ODLink>>();
            #Dictionary<string, GTFSStop> vertices = new Dictionary<string, GTFSStop>();
            links, vertices = GTFSUtils.ExtractRunlinks(dir, AllowedRouteTypes) #NOTE: make sure the Allowed route types is set correctly i.e. (rail,ferry) or (bus,ferry)
            self.vertices = vertices #map of stop code to GTFSFile.GTFSStop object, giving the lat lon

            #now build the graph
            #NOTE: each links.value is a list of the same link, just for different times - we need to average them
            for link in links.values():
                try:
                    Stop_O = vertices[link[0].O]
                    Stop_D = vertices[link[0].D]
                    total_secs = 0
                    count = 0
                    MinSeconds = sys.float_info.max
                    MaxSeconds = 0
                    for timedlink in link:
                        if (timedlink.Seconds < 0):
                            print('PublicTransportNetwork::InitialiseGTFSNetworkGraph: Error, negative runlink time (made positive): ' + timedlink.O + ' ' + timedlink.D + ' ' + timedlink.Seconds)
                            #continue;
                            #you could potentially do abs(timedlink.Seconds) as it looks like they've just been put in the wrong way around?
                            timedlink.Seconds = abs(timedlink.Seconds) #make it positive and continue - needed to do this for gbrail, must have a consistent error which causes an island in the data?
                        #end if
                        total_secs += timedlink.Seconds
                        if (timedlink.Seconds < MinSeconds):
                            MinSeconds = timedlink.Seconds
                        if (timedlink.Seconds > MaxSeconds):
                            MaxSeconds = timedlink.Seconds
                        count+=1
                    #end for timed_link in link
                    AverageSecs = total_secs / count
                    if (AverageSecs < 0):
                        print('Error: negative weight: ' + AverageSecs + ' ' + link[0].O + ' ' + link[0].D)
                    #endif
#set hard minimum limit for runlink to 30 seconds
#                        if ((count==0)||(AverageSecs<30)) AverageSecs = 30.0f;

                    #add an edge (unidirectional as it's an adjacency graph)
                    #TODO: nxgraph code to go here
                    #print('Add edge ',Stop_O.Code,Stop_D.Code)
                    self.graph.add_edge(Stop_O.Code,Stop_D.Code,weight=AverageSecs)
                    #e = WeightedEdge<string>(Stop_O.Code, Stop_D.Code, AverageSecs)
                    #graph.AddVerticesAndEdge(e)
                    #costs.Add(e, AverageSecs)
                    #and add a spatial index entry for finding closest msoa centroids
                    #self.kdtree.Insert(Coordinate(Stop_O.Lon, Stop_O.Lat), Stop_O.Code)
                    #kdtreedata.append([Stop_O.Lon, Stop_O.Lat])
                    #self.kdtree.insert(Stop_O.Code, (Stop_O.Lon, Stop_O.Lat))
                except Exception as e:
                    print('Error: (missing vertex?) ' + link[0].O + ' ' + link[0].D + ' ' + str(e))
                #end try except
            #end for link in links
        #end for dir

        kdtreedata = []
        self.vidx = {}
        for idx, v in enumerate(self.vertices):
            kdtreedata.append([self.vertices[v].Lon,self.vertices[v].Lat])
            self.vidx[idx]=self.vertices[v].Code
        #end for
        #now make the kdtree index in a one shot operation
        print("kdtree length = ",len(kdtreedata))
        self.kdtree = cKDTree(kdtreedata) #NOTE: this is assuming that the node index numbers in networkx graph are the same order as the nodes were created

        print('PublicTransportNetwork::InitialiseGTFSNetworkGraph:: ', self.graph.number_of_nodes(), ' vertices and ', self.graph.number_of_edges(), ' edges in network')

################################################################################

    # <summary>
    # With a bus graph, you might get two bus nodes that are walkable, but aren't linked up because they don't share nodes. This procedure uses the spatial index to find nodes that are close and
    # adds a walking link between then based on a fixed walking speed.
    # PRE: must have called InitialiseGTFSNetworkGraph first.
    # Setting MaxWalkDistance to 2000M results in an additional 3 million edges and that's just for London.
    # 500M seems a better option.
    # TODO: need a way to avoid adding a route between two nodes where there already is a bus link i.e. need to just add walking links between different bus routes, not along bus routes. The route info isn't in here though.
    # TODO: a second alternative might be to route all bus nodes within walking distance to the MSOA centroid which you would have to add as a new node. You still need a stop within 500M, but you could make the radius bigger. No mulitple bus routes though.
    # </summary>
    # <param name="MaxWalkDistance">Maximum walking distance to join up nearby vertices in METRES</param>
    #returns number of edges added
    def FixupWalkingNodes(self,MaxWalkDistance):
        WalkSpeed = 1.39 #This is ms-1, but it's 5kmh or 3.1mph according to Google
        #const float MaxWalkDistance = 2000; //in metres - THIS WAS 500m for buses, changed for rail
        a = 6378137.0 #WGS84 semi-major
        circumference = 2 * math.pi * a
        box = MaxWalkDistance / circumference * 360.0 #search box width based on fraction of earth radius - OK, it's an approximation, so add a bit for safety
        box = box * 1.2

        print('Fixup walking nodes MaxWalkDistance=',MaxWalkDistance,' box=', box)

        # count = 0
        # env = (-180,180,-90,90) #there must be a better way of querying everything?
        # for idx in self.kdtree.query(env):
        #     lat = G.nodes[idx]
        #     lon = G.nodes[idx]
        #     float lat = (float)node.Y, lon = (float)node.X;
        #     foreach (KdNode<string> dnode in kdtree.Query(new Envelope(lon-box,lon+box,lat-box,lat+box))) #the box just needs to cover the MaxWalkDistance, it doesn't matter if it's too big
        #     {
        #         if (node.Data != dnode.Data) #check it's not the same node
        #         {
        #             dist = GTFSUtils.GreatCircleDistance(lat, lon, (float)dnode.Y, (float)dnode.X);
        #             #dist in metres, so set 500M radius around node
        #             if (dist < MaxWalkDistance)
        #             {
        #                 WeightedEdge<string> e = new WeightedEdge<string>(node.Data, dnode.Data, dist/WalkSpeed); #connect two existing nodes with a directed edge - note that we don't add any new vertices, just an edge
        #                 graph.AddEdge(e);
        #                 costs.Add(e, dist / WalkSpeed);
        #                 count+=1
        #             }
        #         }
        #     }
        # }

        #OK, let's do this differently - query all pairs of coordinates that are within the set distance, then go through each and do a better distance test before connecting them up
        #NOTE: I don't think this is anything like as efficient as the original code
        count = 0
        pairs = self.kdtree.query_pairs(box)
        for pair in pairs:
            #print(pair)
            #print(self.kdtree.data[pair[0]],self.kdtree.data[pair[1]])
            #print(self.vidx[pair[0]],self.vidx[pair[1]])
            #print(list(self.graph.nodes)[0])
            #n1 = self.kdtree.data[pair[0]] #NOTE these are (lon,lat)
            #n2 = self.kdtree.data[pair[1]]
            node1 = self.vertices[self.vidx[pair[0]]] #this is a GTFSFile.GTFSStop - NOTE: lat/lon should match with n1,n2
            node2 = self.vertices[self.vidx[pair[1]]]
            #print('n1 n2 n1id n2id',n1,n2,n1id.Code,n1id.Lon,n1id.Lat,n2id.Code,n2id.Lon,n2id.Lat)
            #NOTE: kdtree is made from self.vertices, which came from the gtfs stops. This can have nodes that aren't part of
            #the graph as they haven't been referenced in the gtfs stop_times file which contains the runlinks. The short
            #answer is that we need to check that the node ids are actually in the graph and ignore them if not.
            if node1.Code in self.graph.nodes and node2.Code in self.graph.nodes:
                if pair[0]==pair[1]:
                    print("Error: nodes are the same")
                dist = GTFSUtils.GreatCircleDistance(node1.Lat,node1.Lon,node2.Lat,node2.Lon) #and these are lat,lon
                if (dist<MaxWalkDistance):
                    links = nx.neighbors(self.graph,node1.Code)
                    if not node2 in links: #test whether this node is already connected to the other one
                        self.graph.add_edge(node1.Code, node2.Code, weight = dist/WalkSpeed)
                        count+=1
                        print("FixupWalkingNodes added: ", node1.Code, node2.Code, dist, dist/WalkSpeed)
            #endif


        print('Fixup walking nodes finished. Added ', count, ' new edges.')
            
        return count
    
################################################################################

    # /// <summary>
    # /// As it doesn't work if you try and pick the nearest bus vertex to an msoa centroid, this function finds all the bus vertex points in the msoa and picks a middle one.
    # /// PRE: MUST have called InitialiseGTFSNetworkGraph first to set up the GTFS graph as it uses the spatial index of points (kdtree).
    # /// RWM: NOTE: generalised this from MSOA only to work with any shapefile. This required added a field name for the area code.
    # /// RWM2: NOTE: also changed the code so that it skips and warns of an area in the shapefile not in the zonecodes file - this is for the Manchester and London subsets
    # /// </summary>
    # /// <param name="ZoneCodes"></param>
    # /// <param name="ShapefileFilename"></param>
    # /// <returns>A lookup between the MSOA (or other shapefile geometry e.g. LSOA) areakey and the unqiue string identifier of its calculated centroid bus stop</returns>
    def FindCentroids(self, ZoneCodes, ShapefileFilename, ShapefileAreaKeyField):

        Result = {}
        
        shapefile = gpd.read_file(ShapefileFilename)
        print(shapefile)
        for idx, f in shapefile.iterrows():
            #print(f)
            #for this feature, get the centroid, which we use as the origin
            areakey = f[ShapefileAreaKeyField] #was "MSOA11CD" for MSOA
            #print(f['geometry'].centroid)
            #areaname = f["MSOA11NM"] #not needed
        #     DataRow Rowi = ZoneCodes.Rows.Find(areakey);
        #     if (Rowi==null)
        #     {
        #         #System.Diagnostics.Debug.WriteLine("WARN: area " + areakey + " in shapefile, but not in the zonecodes table - skipped. This will occur when processing subsets of areas from a main shapefile containing every possible area.");
        #         continue;
        #     }
        #     int Zonei = (int)Rowi["Zonei"];
        #     float CentroidLat = (float)Rowi["Lat"];
        #     float CentroidLon = (float)Rowi["Lon"];
        #     List<KdNode<string>> nodes = (List<KdNode<string>>)kdtree.Query(f.Geometry.EnvelopeInternal);
        #     double Lat = 0;
        #     double Lon = 0;
        #     int count = 0;
        #     int MaxOutDegree = 0;
        #     KdNode<string> MaxOutDegreeNode = null;
        #     foreach (KdNode<string> node in nodes)
        #     {
        #         if (f.Geometry.Contains(new Point(node.Coordinate)))
        #         {
        #             Lat += node.Coordinate.Y;
        #             Lon += node.Coordinate.X;
        #             ++count;
        #             #and look at the number of out edges for this node which is within the MSOA in order to find the maximum, which might be a better metric...
        #             int O = graph.OutDegree(node.Data);
        #             if (O > MaxOutDegree)
        #             {
        #                 MaxOutDegree = O;
        #                 MaxOutDegreeNode = node;
        #             }
        #         }
        #     }
        #     if (count > 0)
        #     {
        #         Lat /= count;
        #         Lon /= count;

        #         double minDist2 = double.MaxValue;
        #         KdNode<string> MinNode = null;
        #         foreach (KdNode<string> node in nodes)
        #         {
        #             double dx = node.Coordinate.X - Lon;
        #             double dy = node.Coordinate.Y - Lat;
        #             double dist2 = dx * dx + dy * dy;
        #             if (dist2 < minDist2)
        #             {
        #                 minDist2 = dist2;
        #                 MinNode = node;
        #             }
        #         }

        #         #KdNode<string> node = kdtree.NearestNeighbor(new Coordinate(Lon, Lat)); - DOESN'T WORK! I think the kdtree nearest neighbour function is wrong!
        #         System.Diagnostics.Debug.WriteLine("PublicTransportNetwork::FindCentroids: "+areakey + "," + CentroidLat + "," + CentroidLon + "," + Lat + "," + Lon + "," + MinNode.Coordinate.Y + "," + MinNode.Coordinate.X + "," + MinNode.Data
        #             +","+MaxOutDegree+","+MaxOutDegreeNode.Data+","+count);
        #         #we have options here - either return the closest node to the centroid of the stops within the MSOA (MinNode.data)
        #         #OR return the node within the MSOA with the most out edges (MaxOutDegreeNode)
        #         #Result.Add(areakey, MinNode.Data); //closest to centroid
        #         Result.Add(areakey, MaxOutDegreeNode.Data); //max out edges
        #     }
        #     else
        #     {
        #         #System.Diagnostics.Debug.WriteLine("PublicTransportNetwork::FindCentroids: "+ areakey +", Error"); // + " " + areaname);
        #         System.Diagnostics.Debug.WriteLine("PublicTransportNetwork::FindCentroids: " + areakey + "," + CentroidLat + "," + CentroidLon + "," + Lat + "," + Lon + "," + "0" + "," + "0" + "," + "0"
        #             + "," + "0" + "," + "0" + "," + count);
        #     }    
        #end for f in shapefile
        
        return Result

################################################################################
