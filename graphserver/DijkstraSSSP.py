"""
Test version of a classic optimised Dijkstra algorithm for single source shortest path
"""

import networkx as nx
#from skfmm import heap
#import heapq
import heapdict
#from fibheap import *
import fibonacci_heap_mod

class DijkstraSSSP:
    def __init__(self,G):
        self.graph = G
        self.d = {}
        self.p = {} #contains the best path from u to v
        self.Q = heapdict.heapdict()
        self.fibmap = {} #contains lookup between u,v and Entry in the fib heap for fibonacci_mod_heap

    ################################################################################

    """
    @param s source node in graph
    """
    def dijkstraSSSPTest(self,s):
        self.d={}
        self.d[s]=0
        self.p={}
        self.Q = heapdict.heapdict()
        self.Q[s] = self.d[s]
        for v in self.graph.nodes:
            if v!=s:
                self.d[v]=99999999999 #todo: max float?
                self.p[v]=""
        while len(self.Q)>0:
            u, priority = self.Q.popitem()
            for e in nx.neighbors(self.graph,u):
                self.relax([u,e,0]) #annoyingly, networkx stores edges as triples like this: [origin, dest, ???]

    ################################################################################

    def relax(self,e):
        #get the two edge ends
        #e=[u,v,0] triple to match n
        #print("relax: ",e)
        u=e[0]
        v=e[1]
        w=self.graph.edges[e]['weight']
        #
        if self.d[u]+w<self.d[v]:
            self.d[v]=self.d[u]+w #check - is the right, was c(e), not c(u,v) - it's the same
            if self.p[v]=="":
                self.Q[v]=self.d[v] #This is pushing a new element, v
            else:
                self.Q[v]=self.d[v] #This is a problem, you need a dict heap that lets you modify the priority "decrease-key" and can rebuild the heap to maintain the minimum head
        self.p[v]=u #what are we setting here - this just needs to be a visited flag to prevent cycles?

    ################################################################################

    """
    Prints the results of running dijkstraSSSP i.e. the cost of getting to every node in the graph from the source node
    """
    def debugPrintData(self):
        for v,w in self.d.items():
            if v in self.p:
                pv=self.p[v]
            else:
                pv="n/a"
            print(v,"-->",w, "p[v]=",pv)

    ################################################################################

    """
    Prints the results of running KKP_APSP i.e. the cost of getting to every node in the graph from every other node
    All the d and p hashes are two level now as we've got multiple sources
    """
    def debugPrintAPSPData(self):
        for s in self.d:
            for v,w in self.d[s].items():
                if v in self.p[s]:
                    pv=self.p[s][v]
                else:
                    pv="n/a"
                print(s, "-->", v, "-->" , w , "p[s][v]=",pv)

    ################################################################################

    """
    This is an implementation of the Karger, Koller and Phillips algorithm for APSP using a modification of Dijkstra to speed up finding all the pairs of
    shortest paths simultaneously. 
    D.R. Karger, D. Koller, and S.J. Phillips. Finding the hidden path: time bounds for all-pairs shortest paths. SIAM Journal on Computing, 22:1199–1217, 1993.
    """
    def KKP_APSP(self):
        self.d={} #this is going to be d[u,v] distances
        self.p={} #this is going to be p[u,v] node names
        vin={} #vertex in
        vout={} #vertex out
        for u in self.graph.nodes:
            self.d[u]={} #need to init the hash of hash
            self.d[u][u]=0
            self.p[u]={} #need to init this too
            self.fibmap[u]={} #and this
            vin[u]=[]
            vout[u]=[]
        #endfor
        for u in self.graph.nodes:
            for v in self.graph.nodes:
                if u!=v:
                    if not u in self.d:
                        self.d[u]={}
                        self.p[u]={}
                    #endif
                    self.d[u][v]=99999999999999 #todo: max float
                    self.p[u][v]=""
                    self.fibmap[u][v]=None
                #endif
            #endfor
        #endfor

        #self.Q = heapdict.heapdict()
        #self.Q = makefheap()
        self.Q = fibonacci_heap_mod.Fibonacci_heap()

        for e in self.graph.edges:
            u=e[0]
            v=e[1]
            w=self.graph.edges[e]['weight']
            self.d[u][v]=w
            self.p[u][v]=e
            #self.Q[(u,v)] = self.d[u][v]
            #fheappush(self.Q, (self.d[u][v],(u,v)) )
            entry = self.Q.enqueue( (u,v), self.d[u][v] )
            self.fibmap[u][v]=entry
        #endfor

        while len(self.Q)>0:
        #while self.Q.num_nodes>0:
            #edge, priority = self.Q.popitem()
            #priority, edge = fheappop(self.Q)
            entry = self.Q.dequeue_min()
            edge = entry.m_elem #and weight =m_priority
            #print("weight=",entry.m_priority)

            u = edge[0]
            v = edge[1]
            vin[v].append(u) #insert? TODO
            for e in vout[v]:
                self.KKP_Relax(u,e)
            #endfor
            if self.p[u][v][0]==u: #TODO start[p[u,v]]=u? i.e. start vertex of edge == u
                vout[u].append(self.p[u][v]) #TODO insert?
                for w in vin[u]:
                    self.KKP_Relax(w,self.p[u][v])
                #endfor
            #endif
        #endwhile

    ################################################################################

    """
    Edge relaxation for the Karger, Koller and Phillips algorithm
    """
    def KKP_Relax(self,u,e):
        v=e[0]
        w=e[1]
        weight = self.graph.edges[e]['weight']
        if self.d[u][v]+weight < self.d[u][w]:
            self.d[u][w] = self.d[u][v]+weight
            if self.p[u][w] == "":
                #self.Q[(u,w)]=self.d[u][w] #heap-insert
                #fheappush(self.Q, (self.d[u][w], (u,w)) )
                entry = self.Q.enqueue( (u,w), self.d[u][w] )
                self.fibmap[u][w]=entry
            else:
                #self.Q[(u,w)]=self.d[u][w] #decrease-key
                entry = self.fibmap[u][w]
                self.Q.decrease_key(entry, self.d[u][w])
            self.p[u][w] = e
        #endif

    ################################################################################
