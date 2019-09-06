"""
This is an implementation of the Demetrescu, Italiano and Emiliozzi dynamic Potentially Uniform Paths algorithm (PUP)
http://www.dis.uniroma1.it/demetres/experim/dsp/
"""

###
#from LDSP.h

#this is a library function
class Heap:
    def __init__(self):
        #init

    def extractMin(self):
        #todo

    def insert(self,key,priority):
        #todo

    def decreaseKey(self,key,priority):
        #todo

    def findMin(self):
        #todo


################################################################################

class LDSPPath:
    def __init__(self):
        #init
        #shouldn't be able to call this - MUST use newPathVertex, Edge or Path instead

    

################################################################################


class LDSP:
    def __init__(self):
        #init here!
        self.pi = {}
        self.l=None #TODO: these are wrong!
        self.r=None
        self.start=None
        self.end=None
        self.first=None
        self.last=None
        self.cost=0
        self.sel=False
        self.GL = {}
        self.GR = {}
        self.SL = {}
        self.SR = {}

        self.d = {}
        self.p = {}
        self.P = {}
        self.Q = {}

    ################################################################################

    """
    Build a new path from a single vertex
    @param v The vertex to add
    @returns The new path
    """
    def newPathVertex(self,v):
        #new path from a single vertex
        pi = LDSPPath()
        self.l[pi]=None
        self.r[pi]=None
        self.start[pi]=v
        self.end[pi]=v
        self.first[pi]=None
        self.last[pi]=None
        self.cost[pi]=0
        self.sel[pi]=True
        self.GL[pi] = {}
        self.GR[pi] = {}
        self.SL[pi] = {}
        self.SR[pi] = {}
        return pi
    
    ################################################################################


    """
    Build a new path from an edge
    @param e The edge to add
    @returns The new path
    """
    def newPathEdge(self,e):
        #new path from an edge
        u = e[0]
        v = e[1]
        w = ???
        pi = LDSPPath()
        self.l[pi] = self.pi[u]
        self.r[pi] = self.pi[v]
        self.start[pi] = u
        self.end[pi] = v
        self.first[pi] = e
        self.last[pi] = e
        self.cost[pi] = w
        self.sel[pi] = False
        self.GL[pi] = {}
        self.GR[pi] = {}
        self.SL[pi] = {}
        self.SR[pi] = {}
        #insert TODO
        #insert TODO
        return pi

    ################################################################################

    """
    Build a new path from two existing paths
    @param pi1 First path
    @param pi2 Second path
    @pre r[pi1]==l[pi2], otherwise we throw an error
    @returns The new path
    """
    def newPath(self,pi1,pi2):
        #new path from two existing paths
        if pi1.r!=pi2.l:
            print("LDSPPath newPath(pi1,pi2) ERROR, paths not linked: ",pi1,pi2)
            return None

        pi = LDSPPath()
        pi.l = pi1
        pi.r = pi2
        self.start[pi] = pi1.start
        self.end[pi] = pi2.end
        self.first[pi] = pi1.first
        self.last[pi] = pi2.last
        self.cost[pi] = pi1.cost + pi2.cost #is this right?
        self.sel[pi] = False
        self.GL[pi] = {}
        self.GR[pi] = {}
        self.SL[pi] = {}
        self.SR[pi] = {}
        #insert TODO
        #insert TODO
        return pi

    ################################################################################


    def apsp(self,G):
        #main function, make the shortest paths
        for u in self.graph.nodes:
            self.pi[u] = self.newPathVertex(u)
            self.d[u]={} #need to initialise map of map
            self.d[u][u] = 0
            self.p[u]={} #need to initialise map of map
            self.p[u][u]=self.pi[u]
        #endfor

        #THIS IS POTENTIALLY 1000000 x 1000000 heaps
        for u in self.graph.nodes:
            for v in self.graph.nodes:
                if u!=v:
                    self.P[u] = {} #need to initialise map of map
                    self.P[u][v]
                #endif
            #endfor
        #endfor

        self.insertEdges(G.edges)
        buildPaths(G)

    ################################################################################

    """
    Insert edges into structures
    @param Eins Edges to insert (NetworkX)
    """
    def insertEdges(self,Eins):
        #insert edges
        for e in Eins:
            u = e[0]
            v = e[1]
            pi = self.newpathEdge(e)
            heapInsert(self.P[u,v],pi,cost[pi])

    ################################################################################

    def deleteEdges(self,Edel):
        #delete edges

    ################################################################################

    def buildPaths(self,G):
        #build the shortest paths
        self.Q = heap()
        self.initBuildPaths(G)
        while len(self.Q)>0:
            (u,v) = extractMin(Q)
            newShortestPath(self.p[u][v])
        #endwhile
    
    ################################################################################

    def initBuildPaths(self,G):
        #init
        for 

    ################################################################################

    def newShortestPath(self,pi):
        #new shortest path
        #pi=path
    
    ################################################################################

    def examine(self,pi):
        #examine
        #pi=path

    ################################################################################

    #up to here, the functions are all for static case of apsp
    ################################################################################
    #from here onwards, the functions relate to edge insertions and deletions and weight updates


#from C code
    # PUBLIC FUNCTION PROTOTYPES
    #LDSP*        LDSP_New            (LGraph* inGraph, LEdgeInfo* inEdgeWeights);
    #LDSP*        LDSP_NewEmpty       (ui2 inNumVertices);
    #void         LDSP_Delete         (LDSP** AThis);

    #ui2          LDSP_GetNumVertices (LDSP* This);
    #ui4          LDSP_GetEdgeWeight  (LDSP* This, ui2 inU, ui2 inV);

    #void         LDSP_UpdateEdge     (LDSP* This, ui2 inU, ui2 inV, ui4 inW);
    #ui4          LDSP_GetDist        (LDSP* This, ui2 inX, ui2 inY);

    #ui2          LDSP_GetLWit        (LDSP* This, ui2 inX, ui2 inY);
    #ui2          LDSP_GetRWit        (LDSP* This, ui2 inX, ui2 inY);

    #LDSP_TSetup  LDSP_GetConfig      (LDSP* This);
    #void         LDSP_SetConfig      (LDSP* This, LDSP_TSetup inSetup);
    #ui4          LDSP_GetUsedMem     (LDSP* This);
    #LDSP_TStat   LDSP_GetStatistics  (LDSP* This);
    #void         LDSP_Dump           (LDSP* This);

    