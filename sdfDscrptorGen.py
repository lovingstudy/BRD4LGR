import dist
import pmf_atom_typer as pat
import raw_print_dist as rpd
import resiDescriptorGen_Scaled_LE as rdg
import extract_pdb1 as ep
from pybel import *
import os
from math import exp

E = exp(1.0)
sdf = 'E:/brd4/TPN_opt/position2/SP_4gpj_position2-809_2/SP_4gpj_position2-809_2.sdf'
pr = '4GPJ_confPred.pdb'
#pr = 'e:/brd4/8HQ_opt/318Opt/BRD4_BDF1001_idf_protein.pdb'
expdir = sdf[:-4]+'_BRD4LGR-v4.txt'
expsdf = sdf[:-4]+'_BRD4LGR-v4.sdf'
DESCRIPTOR = rdg.DESCRIPTOR
dc_fn = 'dist_scPDB_train_4.dat'
scalerFit = 'dscrptorValueLE_train.txt'


def brd4_lgr(dd):
    modelfile = './model160622.txt'
    rawPara = [float(l.strip().split()[1]) for l in open(modelfile)]
    rawDES = [l.strip().split()[0] for l in open(modelfile)]
    para = rawPara[:-1]
    descriptor = rawDES[:-1]
    intercept = rawPara[-1]
    t = sum([dd[a]*b for a,b in zip(descriptor, para)])+intercept
    return 1 - 1/(1+E**t)


f = open(expdir, 'w')
ferr = open('errlog.txt','w')
f.write('\t'.join(['Name']+DESCRIPTOR+['Prob'])+'\n')
pts = dist.PTS(dist.UnformatedInput(dc_fn))
fsdout = open(expsdf, 'w')
import time
t = time.time()
scaler = rdg.getScaler_fromFile(scalerFit)
for n,mol in enumerate(readfile('sdf', sdf)):
    try:
        mol.write('sdf', 'ligand%s.sdf'%n)
        ep.main(pr, 'ligand%s.sdf'%n, 'ligand%s_poc.pdb'%n)
        pro = readfile('pdb', 'ligand%s_poc.pdb'%n).next()
        dd = rdg.GetDscrptor(mol, pro, pts)
        entry = [mol.title]+[str(dd[d]) for d in DESCRIPTOR]
        dd = rdg.scaleDES_fromDict(scaler, dd)
        entry += [str(brd4_lgr(dd))]
        f.write('\t'.join(entry)+'\n')
        mol.data['BRD4LGR_pred'] = str(brd4_lgr(dd))
        fsdout.write(mol.write('sdf'))
        os.remove('ligand%s.sdf'%n)
        os.remove('ligand%s_poc.pdb'%n)
        if n%100==0: print n, mol.title, brd4_lgr(dd), time.time()-t
    except:
        ferr.write(str(n)+'\n')

print time.time()-t
f.close()
fsdout.close()
ferr.close()
    
