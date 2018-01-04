import dist
import pmf_atom_typer as pat
import raw_print_dist as rpd
import resiDescriptorGen_Scaled_LE as rdg
import extract_pdb1 as ep
from pybel import *
import os
from math import exp
import mpi4py.MPI as MPI
import time


sdfile = '../'

DESCRIPTOR = rdg.DESCRIPTOR
proString = open('4GPJ_confPred.pdb').read()
pts = dist.PTS(dist.UnformatedInput('dist_scPDB_train_4.dat'))
scaler = rdg.getScaler_fromFile('dscrptorValueLE_train.txt')

def loadModel(modelfile):
    rawPara = [float(l.strip().split()[1]) for l in open(modelfile)]
    rawDES = [l.strip().split()[0] for l in open(modelfile)]
    para = rawPara[:-1]
    descriptor = rawDES[:-1]
    intercept = rawPara[-1]
    return descriptor, para, intercept
model = loadModel('./model160126.txt')

E = exp(1.0)
def brd4_lgr(dd, model):
    descriptor, para, intercept = model
    t = sum([dd[a]*b for a,b in zip(descriptor, para)])+intercept
    return 1 - 1/(1+E**t)

def loadSDF(sdfile, comSize):
    n = 0
    with open(sdfile) as f:
        for l in f:
            if l.startswith('$$$$'): n += 1
    if n%comSize==0: single_size = n/comSize
    else: single_size = n/(comSize-1)
    os.system('echo %s compounds assigned to %s cpus, %s each'%(n,comSize,single_size))
    molPool, singlePool, bufstr = [], [], ''
    with open(sdfile) as f:
        for l in f:
            bufstr += l
            if l.startswith('$$$$'):
                singlePool.append(bufstr)
                bufstr = ''
                if len(singlePool) == single_size:
                    molPool.append(singlePool)
                    singlePool = []
        molPool.append(singlePool)
    return molPool

def singleRun(ligString, proString, pts, model, scaler):
    pocString = ep.Extract(proString, ligString)
    poc = readstring('pdb', pocString)
    mol = readstring('sdf', ligString)
    dd = rdg.GetDscrptor(mol, poc, pts)
    dd = rdg.scaleDES_fromDict(scaler, dd)
    #print [dd[d] for d in DESCRIPTOR[:5]]
    brd4LgrScore = brd4_lgr(dd,model)
    mol.data['BRD4LGR_pred'] = str(brd4LgrScore)
    #entry = [mol.title]+[str(dd[d]) for d in DESCRIPTOR]+[str(brd4LgrScore)]
    return mol.write('sdf')

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

if comm_rank == 0:
    t = time.time()
    os.system('echo -----------------------------------------------------')
    os.system('echo            Scoring work STARTS ...')
    os.system('echo -----------------------------------------------------')
    data = loadSDF(sdfile, comm_size)
    os.system('echo '+str(map(len, data)))
else:
    data = None

local_data = comm.scatter(data, root=0)
os.system('echo rank %d, got and will do: %s'%(comm_rank, len(local_data)))
local_out = []
for i,mol_string in enumerate(local_data):
    mol_scored = singleRun(mol_string, proString, pts, model, scaler)
    local_out.append(mol_scored)
    if i>0 and i%5==0: os.system('echo rank %s scored %s compounds'%(comm_rank, i))

combine_data = comm.gather(local_out,root=0)

if comm_rank == 0:
    os.system('echo'+str(map(len, combine_data)))
    f = open('score.sdf','w')
    for s in combine_data:
        for mol in s:
            f.write(mol)
    os.system('echo -----------------------------------------------------')
    os.system('echo            Scoring work DONE !')
    os.system('echo            Total time: %s s.'%(time.time()-t))
    os.system('echo -----------------------------------------------------')

        
