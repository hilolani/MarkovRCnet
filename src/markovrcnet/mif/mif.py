import os
import numpy as np
import networkx as nx
import math
import itertools
from typing import Optional

from markovrcnet.utils.logging import resolve_logger
from markovrcnet.io import load_adjacency

CONST_COEFFICIENT_ARRAY = (1, 1.618033988749895, 1.8392867552141607, 1.9275619754829254, 1.9659482366454855, 1.9835828434243263, 1.9919641966050354, 1.9960311797354144, 1.9980294702622872, 1.9990186327101014)


def MiF_ZeroBasedIndex(adjacencymatrixchecked, x, y, beta, gamma, logger=None):
    log = resolve_logger(logger, "MiF")
    val = 0
    coefficientlist = list(CONST_COEFFICIENT_ARRAY)
    adj_matrix = adjacencymatrixchecked
    alphalist = [(1 / coefficientlist[gamma]) ** (i + 1) for i in range(0, gamma + 1)]
    i = None
    tmat = {0: adj_matrix}
    for k in range(0, gamma):
        tmat[k + 1] = tmat[k] @ tmat[0]
    for i in range(0, gamma + 1):
        matpower = tmat[i]
        matpowerxandy = matpower[x, y]
        sumupx = np.sum(matpower[x, :])
        sumupy = np.sum(matpower[y, :])
        numerator = alphalist[i] * matpowerxandy * (beta * sumupx + (1 - beta) * sumupy)
        denominator = sumupx * sumupy
        if denominator != 0:
            val += numerator / denominator
    return val

def MiF_OneBasedIndex(adjacencymatrixchecked, x, y, beta, gamma, logger=None):
    log = resolve_logger(logger, "MiF")
    val = 0
    coefficientlist = list(CONST_COEFFICIENT_ARRAY)
    log.info(f"Here, all integer values are assumed to be 1-based indexes, i.e. data and parameters --including node numbers and gamma-- that are counted starting from 1.In other words, it is assumed that a sparse matrix with a 1-based index created in MATLAB, Mathematica, Julia, Fortran, R, etc. was input here.")
    adj_matrix = adjacencymatrixchecked
    alphalist = [(1 / coefficientlist[gamma - 1]) ** (i + 1) for i in range(0, gamma)]
    i = None
    tmat = {0: adj_matrix}
    for k in range(0, gamma - 1):
        tmat[k + 1] = tmat[k] @ tmat[0]
    for i in range(0, gamma):
        matpower = tmat[i]
        matpowerxandy = matpower[x - 1, y - 1]
        sumupx = np.sum(matpower[x - 1, :])
        sumupy = np.sum(matpower[y - 1 , :])
        numerator = alphalist[i] * matpowerxandy * (beta * sumupx + (1 - beta) * sumupy)
        denominator = sumupx * sumupy
        if denominator != 0:
            val += numerator / denominator
    return val

def MiF(adjacencymatrixchecked, x, y, beta, gamma,index_base = 0, gamma_threshold = 10, logger=None):
    log = resolve_logger(logger, "MiF")
    if index_base == 0:
        return MiF_ZeroBasedIndex(adjacencymatrixchecked, x, y, beta, gamma,logger)
    elif index_base == 1:
        return MiF_OneBasedIndex(adjacencymatrixchecked, x, y, beta, gamma,logger)

def MiF_broadcast_withloop(adjacencymatrixchecked, startingvertex, beta = 0.5, gamma_threshold = 10, logger=None):
    np.set_printoptions(precision=10, floatmode='fixed')
    log = resolve_logger(logger, "MiF")
    adj_matrix = adjacencymatrixchecked
    Gobj = nx.from_scipy_sparse_array(adj_matrix)
    degdicformat = nx.degree(Gobj)
    deglst =list([degdicformat[i] for i in range(0,len(degdicformat))])
    log.info(f"the number od nodes: {len(deglst)}")
    alllistednodes = range(0,len(degdicformat))
    if int(startingvertex) > len(deglst):
       log.info(f"The starting node does not exist.")
    elif deglst[startingvertex] == 0:
       log.info(f"The starting node is isolated.")
    else:
       targettednodes = [i for i in alllistednodes if i not in [startingvertex]]
       log.info(f"Starting node to be off target: {startingvertex}")
       log.info(f"List of target nodes: {targettednodes}")
       gammaval = 0
       while gammaval < gamma_threshold:
          mifsteps = [[startingvertex, MiF(adj_matrix, startingvertex, j, beta, gammaval)] for j in targettednodes]
          all_targets = range(len(deglst))  
          other_targets = [t for t in all_targets if t != startingvertex]  
          reached = [[startingvertex, target_id, val] for target_id, (_, val) in zip(other_targets, mifsteps) if val != 0.0]
          log.info(f"The  number of reached nodes: {len(reached)}")
          log.info(f"Reached vertices information: {reached}")
          gammaval = gammaval + 1
          print(f"gammaval: {gammaval}")
          if len(mifsteps) == len(reached):
             log.info(f"Gamma reached the maximum values, since all the nodes have been reached from the starting nodes.")
             return reached
             break

def MiF_broadcast_withoutloop(adjacencymatrixchecked, startingvertex, beta = 0.5, gamma_threshold = 10, logger=None):
    log = resolve_logger(logger, "MiF")
    adj_matrix = adjacencymatrixchecked
    Gobj = nx.from_scipy_sparse_array(adj_matrix)
    degdicformat = nx.degree(Gobj)
    deglst =list([degdicformat[i] for i in range(0,len(degdicformat))])
    log.info(f"the number od nodes: {len(deglst)}")
    alllistednodes = range(0,len(degdicformat))
    if int(startingvertex) > len(deglst):
       log.info(f"The starting node does not exist.")
    elif deglst[startingvertex] == 0:
       log.info(f"The starting node is isolated.")
    else:
       targettednodes = [i for i in alllistednodes if i not in [startingvertex]]
       gammaval = 0
       reachedlist =[]
       reachedinfolist =[]#new
       reachednodevalslist = []
       remainingnodes = targettednodes
       while gammaval < gamma_threshold:
           reachednodesfromstartingnodes = []
           reachednodesfromeachstartingnode = []
           mifsteps = [[startingvertex, j, MiF(adj_matrix, startingvertex, j, beta, gammaval)] for j in remainingnodes]
           reached = [x for i, x in enumerate(mifsteps) if x[2]!= np.float64(0.0)]
           log.info(f"reached: {reached}")
           reachedinfolist.append(reached)
           reachednodes = [x[1] for i, x in enumerate(reached)]
           log.info(f"list of nodes that, once reached, should be skipped without going any further, i.e. reachednodes: {reachednodes}")
           reachedlist.append(reachednodes)
           log.info(f"reachedlist: {reachedlist}")
           log.info(f"reachedinfolist: {reachedinfolist}")
           remainingnodes = [x for i, x in enumerate(remainingnodes) if x not in list(itertools.chain.from_iterable(reachedlist))]
           log.info(f"list of nodes that remain to be reached, i.e. remainingnodes: {remainingnodes}")
           gammaval += 1
           reachedlist = []
           reachednodevalslist = reachednodevalslist + reached
           if len(remainingnodes) == 0:
               log.info(f"Gamma reached the maximum values, since all the nodes have been reached from the starting nodes.")
               finalreachedresult = sorted([item for sublist in reachedinfolist for item in sublist], key=lambda x: x[1])
               return finalreachedresult
               break

def MiF_broadcast(adjacencymatrixchecked, startingvertex, beta = 0.5, gamma_threshold = 10, loop = 0,logger=None):
    log = resolve_logger(logger, "MiF")
    if loop == 0:
        return MiF_broadcast_withoutloop(adjacencymatrixchecked, startingvertex, beta, gamma_threshold,logger)
    elif loop == 1:
        return MiF_broadcast_withloop(adjacencymatrixchecked, startingvertex, beta, gamma_threshold,logger)

def MiF_broadcast_diff_on_loop(result_withloop, result_withoutloop, logger=None):
    log = resolve_logger(logger, "MiF")
    diff_positions = []
    for i, (row0, row1) in enumerate(zip(result_withloop, result_withoutloop)):
        if row0[:2] != row1[:2]:
            log.error(f"The key differs in index {i}. No symmetrical comparison is possible.")
            continue
        if not np.isclose(row0[2], row1[2]):
            diff_positions.append({
                'vertex pair': row0[:2],
                'withloop_value': row0[2],
                'withoutloop_value': row1[2],
                "value diffrerence": row0[2] - row1[2]
            })
    log.info(f"The list of difference betweeen MiF_broadcast with and without loop: {diff_positions}")    
    return diff_positions
    
def MiFDI_withloop(adjacencymatrixchecked, startingvertices = "min", dangn = 0, beta = 0.2, gamma_threshold = 10, allstartinginfo = 0, logger=None):
  log = resolve_logger(logger, "MiF")
  adj_matrix = adjacencymatrixchecked
  Gobj = nx.from_scipy_sparse_array(adj_matrix)
  degdicformat = nx.degree(Gobj)
  deglst =list([degdicformat[i] for i in range(0,len(degdicformat))])
  alllistednodes = range(0,len(degdicformat))
  if startingvertices == "min":
      smallestdegval = min(deglst)
      mindegnodes = [i for i, x in enumerate(deglst) if x == min(deglst)]
      log.info(f"the smallest degree: {smallestdegval}")
      log.info(f"the node numbers with the smallest degree : {mindegnodes}")
      startingnodes = mindegnodes
  elif startingvertices == "max":
      largestdegval = max(deglst)
      maxdegnodes = [i for i, x in enumerate(deglst) if x == max(deglst)]
      log.info(f"the largest degree: {largestdegval}")
      log.info(f"the node numbers with the largest degree : {maxdegnodes}")
      startingnodes = maxdegnodes
  if dangn > 0 and dangn <=len(startingnodes):
     log.info(f"There are multiple nodes for starting: {startingnodes}, so you can choose one starting node to be focused.")
  elif dangn == 0:
     log.info(f"There is only a single node for starting: {startingnodes}")
  else:
     msg = f"the dangn value exceeds the number of the starting nodes."
     log.error(msg)
     raise TypeError(msg)
  gammaval = 0
  logmifmeanlist =[]
  focusedstarting = startingnodes[dangn]
  log.info(f"The focused starting node is: {focusedstarting}")
  all_targets = range(len(deglst))
  allresultlist = []
  allresultlist_i = []
  mifdilist = []
  for i in range(0, len(startingnodes)):
    gammaval = 0
    while gammaval < gamma_threshold:
         other_targets = [t for t in all_targets if t != startingnodes[i]]
         log.info(f"Starting node of this step {gammaval}: {startingnodes[i]}")
         log.info(f"Othertargest for the starting node {startingnodes[i]}:{other_targets}")
         mifsteps = [[startingnodes[i], MiF(adj_matrix, startingnodes[i], j, beta, gammaval)] for j in other_targets]
         log.info(f"MiF steps for the starting node {startingnodes[i]}: {mifsteps}")
         log.info(f"The  number of MiF steps for the starting node {startingnodes[i]}: {len(mifsteps)}")
         reached = [[startingnodes[i], target_id, val] for target_id, (_, val) in zip(other_targets, mifsteps) if val != 0.0]
         log.info(f"reached nodes for the starting node {startingnodes[i]}: {reached}")
         log.info(f"The  number of reached nodes for the starting node {startingnodes[i]}: {len(reached)}")
         logresultinfo_tmp =  [[i[0], i[1], math.log(i[2])] for i in reached]
         log.info(f"Current gamma for the starting node {startingnodes[i]}: {gammaval}, [Starting node, Reached node, Log(MiF)]: {logresultinfo_tmp}")
         logresultinfo = [[x[0], x[1] - startingnodes.index(x[0]) *  len(other_targets), x[2]] for i, x in enumerate(logresultinfo_tmp)]
         meanlog =  np.mean([logresultinfo[l][2] for l in range(len(logresultinfo))])
         log.info(f"Current gamma for the starting node {startingnodes[i]}: {gammaval}, Mean of the Log(MiF): {meanlog} for the starting node {startingnodes[i]}")
         logmifmeanlist.append(meanlog)
         allresult_i = sorted(logresultinfo + [[startingnodes[i], startingnodes[i], 0.0]],  key=lambda x: x[1])
         allresultlist_i = allresultlist_i + allresult_i
         allresultlist_i = [list(t) for t in list(dict.fromkeys(tuple(row) for row in sorted(allresultlist_i)))]
         log.info(f"semifinal allresult for {startingnodes[i]}: {allresultlist_i}")
         
         if len(mifsteps) == len(reached):
             log.info(f"Gamma reached the maximum values, since all the nodes have been reached from the starting node {startingnodes[i]}.")
             allresultlist_i = allresultlist_i + allresult_i
             allresultlist_i = [list(t) for t in list(dict.fromkeys(tuple(row) for row in sorted(allresultlist_i)))]
             allresult_i = sorted(logresultinfo_tmp + [[startingnodes[i], startingnodes[i], 0.0]],  key=lambda x: x[1])
             log.info(f"final allresult for {startingnodes[i]}: {allresult_i}")
             mifval_i = [x[2] for i, x in enumerate(allresult_i)]
             log.info(f"mifval_i : {mifval_i}")
             log.info(f"final MiFDI value for {startingnodes[i]}: {mifval_i} for the starting node {startingnodes[i]}.")
             allresultlist = allresultlist + [allresult_i]
             mifdilist = mifdilist + [mifval_i]
             i = i + 1
             break
         gammaval = gammaval + 1
  if dangn ==0 or dangn <=len(startingnodes):
      if allstartinginfo == 0:
          return allresultlist[dangn], mifdilist[dangn]
      else:    
          return allresultlist, mifdilist
  else:
     msg = f"the dangn value has something wrong."
     log.error(msg)
     raise TypeError(msg)    

def MiFDI_withoutloop(adjacencymatrixchecked, startingvertices = "min", dangn = 0, beta = 0.2, gamma_threshold = 10, allstartinginfo = 0, logger=None):
  log = resolve_logger(logger, "MiF")
  adj_matrix = adjacencymatrixchecked
  Gobj = nx.from_scipy_sparse_array(adj_matrix)
  degdicformat = nx.degree(Gobj)
  deglst =list([degdicformat[i] for i in range(0,len(degdicformat))])
  alllistednodes = range(0,len(degdicformat))
  if startingvertices == "min":
      smallestdegval = min(deglst)
      mindegnodes = [i for i, x in enumerate(deglst) if x == min(deglst)]
      log.info(f"the smallest degree: {smallestdegval}")
      log.info(f"the node numbers with the smallest degree : {mindegnodes}")
      startingnodes = mindegnodes
  elif startingvertices == "max":
      largestdegval = max(deglst)
      maxdegnodes = [i for i, x in enumerate(deglst) if x == max(deglst)]
      log.info(f"the largest degree: {largestdegval}")
      log.info(f"the node numbers with the largest degree : {maxdegnodes}")
      startingnodes = maxdegnodes
  if dangn > 0 and dangn <=len(startingnodes):
     log.info(f"There are multiple nodes for starting: {startingnodes}, so you can choose one starting node to be focused.")
  elif dangn == 0:
     log.info(f"There is only a single node for starting: {startingnodes}")
  else:
     msg = f"the dangn value exceeds the number of the starting nodes."
     log.error(msg)
     raise TypeError(msg)
  gammaval = 0
  logmifmeanlist =[]
  focusedstarting = startingnodes[dangn]
  log.info(f"The focused starting node is: {focusedstarting}")
  all_targets = range(len(deglst))
  allresultlist = []
  mifdilist = []
  for i in range(0, len(startingnodes)):
    gammaval = 0
    allresultlist_i = []
    allresult_i = []
    mifdi_i = []
    logmiflist =[]
    reachedlist =[]
    reachednodesfromeachstartingnode = []
    other_targets = [t for t in all_targets if t != startingnodes[i]]
    remainingnodes = other_targets
    while gammaval < gamma_threshold:
         log.info(f"Starting node of this step {gammaval}: {startingnodes[i]}")
         log.info(f"Remaining target of this step {gammaval} for the starting node {startingnodes[i]}: {remainingnodes}")
         mifsteps = [[startingnodes[i], j, MiF(adj_matrix, startingnodes[i], j, beta, gammaval)] for j in remainingnodes]
         log.info(f"MiF steps for the starting node {startingnodes[i]}: {mifsteps}")
         log.info(f"The  number of MiF steps for the starting node {startingnodes[i]}: {len(mifsteps)}")
         reached = [x for i, x in enumerate(mifsteps) if x[2]!= np.float64(0.0)]
         log.info(f"reached nodes for the starting node {startingnodes[i]}: {reached}")
         log.info(f"The  number of reached nodes for the starting node {startingnodes[i]}: {len(reached)}")
         logresultinfo =  [[i[0], i[1], math.log(i[2])] for i in reached]
         log.info(f"Current gamma: {gammaval}, [Starting node, Reached node, Log(MiF)]: {logresultinfo}")
         logvals = [x[2] for i, x in enumerate(logresultinfo)]
         logmiflist.append(logvals)
         reachednodes = [x[1] for i, x in enumerate(logresultinfo)]
         reachedlist = sorted(reachedlist + reachednodes)
         log.info(f"reachedlist:{reachedlist}")
         reachednodesfromeachstartingnode = reachednodesfromeachstartingnode + reachedlist
         reachednodesfromeachstartingnode = sorted(set(reachednodesfromeachstartingnode))
         log.info(f"list of nodes that, once reached, should be skipped without going any further, i.e. reachednodesskip: {reachednodesfromeachstartingnode}")
         remainingnodes = [x for i, x in enumerate(remainingnodes) if x not in reachednodesfromeachstartingnode]
         log.info(f"list of nodes that remain to be reached, i.e. remainingnodes: {remainingnodes}")
         allresult_i = sorted(logresultinfo + [[startingnodes[i], startingnodes[i], 0.0]],  key=lambda x: x[1])
         allresultlist_i = allresultlist_i + allresult_i
         allresultlist_i = [list(t) for t in list(dict.fromkeys(tuple(row) for row in sorted(allresultlist_i)))]
         log.info(f"semifinal allresult for {startingnodes[i]}: {allresultlist_i}")
         if len(mifsteps) == len(reached):
             log.info(f"Gamma reached the maximum values, since all the nodes have been reached from the starting node {startingnodes[i]}.")
             allresultlist_i = allresultlist_i + allresult_i
             allresultlist_i = [list(t) for t in list(dict.fromkeys(tuple(row) for row in sorted(allresultlist_i)))]
             log.info(f"final allresult for {startingnodes[i]}: {allresultlist_i}")
             mifval_i = [x[2] for i, x in enumerate(allresult_i)]
             log.info(f"mifval_i : {mifval_i}")
             log.info(f"final MiFDI value for {startingnodes[i]}: {mifval_i} for the starting node {startingnodes[i]}.")
             allresultlist = allresultlist + [allresultlist_i]
             mifdilist = mifdilist + [mifval_i]
             log.info(f"final allresult for {startingnodes}: {allresultlist}")
             i = i + 1
             break
         gammaval = gammaval + 1
         allresultlist =[row for row in allresultlist]
         mifdilist = [[inner[2] for inner in group] for group in allresultlist]
  if dangn ==0 or dangn <=len(startingnodes):
      if allstartinginfo == 0:
          return allresultlist[dangn], mifdilist[dangn]
      else:    
          return allresultlist, mifdilist
  else:
     msg = f"the dangn value has something wrong."
     log.error(msg)
     raise TypeError(msg)    

      
def MiFDI(adjacencymatrixchecked, startingvertices="min", dangn = 0, beta = 0.2, gamma_threshold = 10, allstartinginfo = 0, loop = 0, logger=None):
    log = resolve_logger(logger, "MiF")
    if loop == 0:
        return MiFDI_withoutloop(adjacencymatrixchecked, startingvertices, dangn, beta, gamma_threshold, allstartinginfo, logger)
    elif loop == 1:
        return MiFDI_withloop(adjacencymatrixchecked, startingvertices, dangn, beta, gamma_threshold, allstartinginfo, logger)

def MiFDI_diff_on_loop(*args, **kwargs):
    return MiF_broadcast_diff_on_loop(*args, **kwargs)

