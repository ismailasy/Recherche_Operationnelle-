importation des differentes bobliotheque utiliser dans notre projet
import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
from heapq import *
from collections import *

#Definition des fonction des differents algorithme utilisé dans le projet

#Fonction qui trace le graphe
def tracer_graphe(matrice):

    G = nx.DiGraph(matrice)
    st.pyplot(nx.draw(G,node_size=500))

#ford Fulkerson 
class Graph:
    
    def __init__(self, graph):
        self.graph = graph
        self. ROW = len(graph)


    def searching_algo_BFS(self, s, t, parent):

        visited = [False] * (self.ROW)
        queue = []

        queue.append(s)
        visited[s] = True

        while queue:

            u = queue.pop(0)

            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        return True if visited[t] else False

    def ford_fulkerson(self, source, sink):
        parent = [-1] * (self.ROW)
        max_flow = 0

        while self.searching_algo_BFS(source, sink, parent):

            path_flow = float("Inf")
            s = sink
            while(s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            max_flow += path_flow

    
            v = sink
            while(v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]
                

        return max_flow


#Fonction Djikstra
def dijkstra(M, s):
    inf = sum(sum(ligne) for ligne in M) + 1
    nb_sommets = len(M)
    s_explore = {s : [0, [s]]}
    #On associe au sommet d'origine s la liste [longueur, plus court chemin]
    s_a_explorer = {j : [inf, ""] for j in range(nb_sommets) if j != s}
    #On associe à chaque sommet j à explorer la liste [longueur, sommet précédent]
    for suivant in range(nb_sommets):
        if M[s][suivant]:
            s_a_explorer[suivant] = [M[s][suivant], s]

    print("Dans le graphe d\'origine {} de matrice d\'adjacence :".format(s))
    for ligne in M:
       print(ligne)
    print()
    st.write("Plus courts chemins de")

    while s_a_explorer and any(s_a_explorer[k][0] < inf for k in s_a_explorer):
        s_min = min(s_a_explorer, key = s_a_explorer.get)
        longueur_s_min, precedent_s_min = s_a_explorer[s_min]
        for successeur in range(nb_sommets):
            if M[s_min][successeur] and successeur in s_a_explorer:
                dist = longueur_s_min + M[s_min][successeur]
                if dist < s_a_explorer[successeur][0]:
                    s_a_explorer[successeur] = [dist, s_min]
        s_explore[s_min] = [longueur_s_min, s_explore[precedent_s_min][1] + [s_min]]
        del s_a_explorer[s_min]
        st.write("longueur", longueur_s_min, ":", " -> ".join(map(str, s_explore[s_min][1])))

    for k in s_a_explorer:
        st.write("Il n\'y a pas de chemin de {} à {}".format(s, k))

    return s_explore
#Fonction qui lit le fichier ou excel
def lire_fichier():
    file = st.file_uploader("choisir un fichier de la matrice noeud a noeud ",type =['csv','xlsx'])
    status_fichier = st.empty()
    if not file :
        status_fichier.info("S'il vous plait uploder un fichier")
        return
    #type_fichier = get_file_type(file)
    try:
        data = pd.read_csv(file)
        donnes = data[data.columns[:-1]]
        st.dataframe(donnes)
    except :
        data = pd.read_excel(file)
        donnes = data
        st.dataframe(donnes)        

    matrice_adj = np.array(donnes)
    return matrice_adj

#Fonction algorithme de Prim
def Prim (Graphe):
    T = []
    n = len(Graphe)
    st.write("n=",n)
    plusProche = []
    distanceMin = []
    
    for i in range(0,n):
        plusProche.append(0)
        distanceMin.append(0)
    
    for i in range(1,n):
        plusProche[i] = 0
        distanceMin[i] = Graphe[i][0]
    
    for i in range(0,n-1):
        min = 0
        for j in range(1,n):
            if ((min and distanceMin[j] and 0 <= distanceMin[j] < min) or (not min and  0 <= distanceMin[j])):
                min = distanceMin[j]
                k = j
        T.append((k, plusProche[k]))
        #st.write(T)
    
        distanceMin[k] = -1
        distanceMin[plusProche[k]] = -1
    
        for j in range(1,n):
            if ((distanceMin[j] and Graphe[k][j] and Graphe[k][j] < distanceMin[j]) or not distanceMin[j] ):
                distanceMin[j] = Graphe[k][j]
                distanceMin[k] = Graphe[j][k]
            
                plusProche[j] = k
                plusProche[k] = j     
    return T

#Les fonction de Kruskal
def find(i):
    while parent[i] != i:
	    i = parent[i]
    return i

def union(i, j):
    a = find(i)
    b = find(j)
    parent[a] = b

def kruskalMST(cost):
    mincost = 0 

    for i in range(V):
	    parent[i] = i

    edge_count = 0
    while edge_count < V - 1:
        min = INF
        a = -1
        b = -1
        for i in range(V):
            for j in range(V):
                if find(i) != find(j) and cost[i][j] < min:
                    min = cost[i][j]
                    a = i
                    b = j
        union(a, b)
        st.write('Arc {}:({}, {}) cout:{}'.format(edge_count, a, b, min))
        edge_count += 1
        mincost += min
    st.write("Cout Minimun= {}".format(mincost))


if __name__ == '__main__':
    
    #tracer_graphe(donnes) 
    st.title("Projet Recherche operationnel M2SR2021") 
    st.header("Presente par : El Hadj Ibrahima TRAORE Ismaila SY  Fatou NDONG ")
    donnes = lire_fichier()
    choix = st.sidebar.selectbox("Merci de choisir ce que vous vouler faire avec le graphe",['Afficher graphe','derouler algorithme graphe'])
    if choix == "Afficher graphe":
        tracer_graphe(donnes)
    else:
        algo =st.sidebar.selectbox("Choisser un algorithme",['Djikstra','Ford Fulkerson','Kruskal','Prim'])
        
        if algo =='Djikstra':
            debut = st.sidebar.text_input("Debut", key="debut")
            #fin = st.sidebar.text_input("Fin", key="fin")
            deb = int(debut)
            #fin =int(fin)
            d =dijkstra(donnes,deb)
            #st.write("le plus court chemin est :")
            #for i in range(len(d)):
                #st.write(d[i])
        
        
        elif algo =='Ford Fulkerson':
            source = st.sidebar.text_input("Source", key="source")
            puits = st.sidebar.text_input("Puits", key="puit")
            st.write("source :",source,"destination:",puits)
            sour = int(source)
            p = int(puits)
            g = Graph(donnes)
            st.write("le flot max qu'on peut transporte dans ce reseau est : %d " % g.ford_fulkerson(sour, p))        
        elif algo == 'Kruskal':
            #st.write("Merci Kruskal")
            V = len(donnes)
            INF = float('inf')
            donnes = donnes.astype(float)
            for k in range(V):
                for l in range(V):
                    if donnes [k][l] ==0:
                        donnes [k][l] =INF
            parent = [i for i in range(V)]
            st.write("le chemin amiltonient obtenu avec l'algorithme de Kruskal est :")
            k= kruskalMST(donnes)
        
        else:
           # st.write("l'arbre couvrant minimum est :")
            X= Prim(donnes)
            st.write("l'arbre couvrant minimum est :")
            for i in range(len(X)):
                st.write(X[i])
            
            

        

