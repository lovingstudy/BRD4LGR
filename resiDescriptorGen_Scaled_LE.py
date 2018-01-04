#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      user
#
# Created:     09/04/2011
# Copyright:   (c) user 2011
# Licence:     <your licence>
#
# Modified by Yolanda on Apr 28, 2015. Calculate sum of pair potential of some
#     residues as descriptors.
#-------------------------------------------------------------------------------
import pybel
import dist
import pmf_atom_typer as pat
#import simple_atom_typer as pat
import raw_print_dist as rpd
from math import floor
import os
import numpy as np
from sklearn.preprocessing import RobustScaler


DESCRIPTOR = ['140_np','140_polar','146_np','146_polar','81_np','81_polar','82_np','82_polar',\
              '83_np','83_polar','85_np','85_polar','87_np','87_polar','92_np','92_polar',\
              '94_np','94_polar','97_np','97_polar','140_hb','electro','w_hb']

def GetDscrptor(lig,pro,pts):
	dscrptorDict1 = dict(zip(DESCRIPTOR, [0.0]*len(DESCRIPTOR)))
	cutoff=12.0
	polar_type = ['CP','AP','CW','CO','CN','NC','NP','NA','ND','NR','N0',\
                      'NS','OC','OA','OE','OS','OD','SO','P','SA','SD','CL',\
                      'F','Br','I','ME','OW','Se']
	np_type = ['CF','AF','C3']
	chg_type = ['NC','OC']

	def getScore(d, a1_type, a2_type):
		k=int((d-dist.D_MIN*0.1)/dist.D_DELT*10.0)
		dis_type='%s-%s'%(a1_type,a2_type)
		if dis_type not in dist.DC_TYPES: return 0.0
		j=dist.DC_TYPES.index(dis_type)
		nj_bulk = pts.vect[j][0][3]
		if nj_bulk<1000: return 0.0
		return pts.vect[j][k][0]
	def form_hbond(d, a1_type, a2_type):
		if d > 3.2: return False
		hb_a = ['OA','OW','OE','OS','OC','NA','N0','NR','NP','SA','SO','F']
		hb_d = ['OD','OW','NS','ND','NC','SD']
		if a1_type in hb_a and a2_type in hb_d: return True
		elif a1_type in hb_d and a2_type in hb_a: return True
		return False
	def form_electro(d, a1_type, a2_type):
		if d > 12.0: return False
		if a1_type in chg_type and a2_type in chg_type: return True
		return False
	def form_polar(d, a1_type, a2_type):
		if form_hbond(d, a1_type, a2_type): return False
		if form_electro(d, a1_type, a2_type): return False
		if d > 9.0: return False
		if a1_type in polar_type and a2_type in polar_type: return True
		return False
	def form_np(d, a1_type, a2_type):
		return all([d<6.0, a1_type in np_type, a2_type in np_type])
	def dTypeGen_old(alist, blist):
		for a in alist:
			for b in blist:
				yield rpd.GetDistance(a.OBAtom, b.OBAtom),pat.LigAtomTyper(a.OBAtom),pat.ProAtomTyper(b.OBAtom)
	def dTypeGen(alist, blist, form):
		sl, sr, dtlist = set(), set(), []
		for a in alist:
			for b in blist:
				d, a1_type, a2_type = rpd.GetDistance(a.OBAtom, b.OBAtom),pat.LigAtomTyper(a.OBAtom),pat.ProAtomTyper(b.OBAtom)
				if form(d, a1_type, a2_type):
					sl.add(a)
					sr.add(b)
					dtlist.append((d, a1_type, a2_type))
		return dtlist, len(sl), len(sr)
	def keyResi(num): return [a2 for a2 in heavyProAtoms if a2.OBAtom.GetResidue().GetNum()==num]

	
	# Functional programming: more independent energy decomposition
	heavyLigAtoms = [a1 for a1 in lig.atoms if not a1.OBAtom.IsHydrogen()]
	heavyProAtoms = [a2 for a2 in pro.atoms if not a2.OBAtom.IsHydrogen()]
	waterOs = [a2 for a2 in heavyProAtoms if pat.ProAtomTyper(a2.OBAtom) == 'OW']
	
	# Water interactions
	dist_type_list_whb, w_hbNl, w_hbNr = dTypeGen(heavyLigAtoms, waterOs, form_hbond)
	if w_hbNl != 0.0 and w_hbNr != 0.0:
		score_list_whb = [getScore(*dt) for dt in dist_type_list_whb]
		dscrptorDict1['w_hb'] += sum(score_list_whb) / w_hbNl
	
	# 140 H-bond interactions
	dist_type_list_140hb, hb140Nl, hb140Nr = dTypeGen(heavyLigAtoms, keyResi(140), form_hbond)
	if hb140Nl != 0.0 and hb140Nr != 0.0:
		score_list_140hb = [getScore(*dt) for dt in dist_type_list_140hb]
		dscrptorDict1['140_hb'] += sum(score_list_140hb) / hb140Nl
	
	# Electro interactions
	dist_type_list_e, eNl, eNr = dTypeGen(heavyLigAtoms, heavyProAtoms, form_electro)
	if eNl != 0.0 and eNr != 0.0:
		dscrptorDict1['electro'] += sum([getScore(*dt) for dt in dist_type_list_e]) / eNl
	
	# Others: polar or non-polar interactions
	for des in DESCRIPTOR[:-3]:
		num = int(des.split('_')[0])
		if des.endswith('np'):
			dist_type_list_np, npNl, npNr = dTypeGen(heavyLigAtoms, keyResi(num), form_np)
			if npNl != 0.0 and npNr != 0.0:
				score_list_np = [getScore(*dt) for dt in dist_type_list_np]
				dscrptorDict1[des] += sum(score_list_np) / npNl
		elif des.endswith('polar'):
			dist_type_list_polar, polarNl, polarNr = dTypeGen(heavyLigAtoms, keyResi(num), form_polar)
			if polarNl != 0.0 and polarNr != 0.0:
				score_list_polar = [getScore(*dt) for dt in dist_type_list_polar]
				dscrptorDict1[des] += sum(score_list_polar) / polarNl

	return dscrptorDict1


def scaleDES(dscrptorTable, trainScaler):
	X = np.array(dscrptorTable)
	if trainScaler == None:
		scaler = RobustScaler()
		Xt = scaler.fit_transform(X)
		return list(Xt), scaler
	else:
		Xt = trainScaler.transform(X)
		return list(Xt), trainScaler

def getScaler_fromFile(trainFile):
	raw_train = [l.strip().split('\t')[2:] for l in open(trainFile)][1:]
	trainTable = [[float(i) for i in r] for r in raw_train]
	X = np.array(trainTable)
	scaler = RobustScaler()
	scaler.fit(X)
	return scaler
	
def scaleDES_fromFile(scaler, inputFile):
	raw = [l.strip().split('\t') for l in open(inputFile)][1:]
	inputTable = [map(float,r[2:]) for r in raw]
	actv = [r[:2] for r in raw]
	X = np.array(inputTable)
	outTable = [a+list(b) for a,b in zip(actv, scaler.transform(X))]
	return outTable
	
def scaleDES_fromDict(scaler, inputDict):
	X = np.array([[inputDict[d] for d in DESCRIPTOR]])
	outTable = list(scaler.transform(X)[0])
	outDict = dict([(DES,des) for DES,des in zip(DESCRIPTOR, outTable)])
	return outDict
	


#--------------------------------- Output Functions ------------------------------------------------------#
def singleTest(dc_fn):
	lig = pybel.readfile('sdf', 'E:/brd4/PMF_Yolanda/dataSet/version2/trainsetLigand/PDB_3U5L.sdf').next()
	pro = pybel.readfile('pdb', 'E:/brd4/PMF_Yolanda/dataSet/version2/trainsetPocket/PDB_3U5L.pdb').next()
	dc = dist.UnformatedInput(dc_fn)
	pts = dist.PTS(dc)
	dd = GetDscrptor(lig, pro, pts)
	dscrptors = [str(dd[d]) for d in DESCRIPTOR]
	for i in DESCRIPTOR: print i, dd[i]
	return dscrptors

def dscrptorOut(dc_fn):
	dirs = {'Train':{'lig':'e:/brd4/PMF_Yolanda/dataSet/version4/trainsetLigand/',\
					'pro':'e:/brd4/PMF_Yolanda/dataSet/version4/trainsetPocket/',\
					'exp':'e:/brd4/PMF_Yolanda/dataSet/version4/dscrptorValueLE_train.txt',\
					'exp_s':'e:/brd4/PMF_Yolanda/dataSet/version4/dscrptorValueLEScaled_train.txt'},\
			'Test':{'lig':'e:/brd4/PMF_Yolanda/dataSet/version4/testsetLigand/',\
					'pro':'e:/brd4/PMF_Yolanda/dataSet/version4/testsetPocket/',\
					'exp':'e:/brd4/PMF_Yolanda/dataSet/version4/dscrptorValueLE_test.txt',\
					'exp_s':'e:/brd4/PMF_Yolanda/dataSet/version4/dscrptorValueLEScaled_test.txt'}
			}
	dc = dist.UnformatedInput(dc_fn)
	pts = dist.PTS(dc)
	
	def writeDES(ligdir, prodir, expdir):
		with open(expdir, 'w') as f:
			f.write('\t'.join(['Name','Active']+DESCRIPTOR)+'\n')
			for num,p in enumerate(os.listdir(ligdir)):
				print p, num
				a = 0
				lig = pybel.readfile('sdf', ligdir+p).next()
				pro = pybel.readfile('pdb', prodir+p[:-4]+'.pdb').next()
				if lig.data['Active'] == '1': a = 1
				dd = GetDscrptor(lig, pro, pts)
				dscrptors = [str(dd[d]) for d in DESCRIPTOR]
				f.write(p[:-4].replace('ss','/')+'\t'+str(a)+'\t'+'\t'.join(dscrptors)+'\n')
	
	writeDES(dirs['Train']['lig'], dirs['Train']['pro'], dirs['Train']['exp'])
	writeDES(dirs['Test']['lig'], dirs['Test']['pro'], dirs['Test']['exp'])
	scaler = getScaler_fromFile(dirs['Train']['exp'])
	table_train = scaleDES_fromFile(scaler, dirs['Train']['exp'])
	table_test = scaleDES_fromFile(scaler, dirs['Test']['exp'])
	with open(dirs['Train']['exp_s'], 'w') as ftrain:
		ftrain.write('\t'.join(['Name','Active']+DESCRIPTOR)+'\n')
		for i in table_train:
			ftrain.write('\t'.join(map(str,i))+'\n')
	with open(dirs['Test']['exp_s'], 'w') as ftest:
		ftest.write('\t'.join(['Name','Active']+DESCRIPTOR)+'\n')
		for i in table_test:
			ftest.write('\t'.join(map(str,i))+'\n')
	

def dscrptorOut_robscale(dc_fn):
	dirs = {'Train':{'lig':'e:/brd4/PMF_Yolanda/dataSet/version3/trainsetLigand/',\
					'pro':'e:/brd4/PMF_Yolanda/dataSet/version3/trainsetPocket/',\
					'exp':'e:/brd4/PMF_Yolanda/dataSet/version3/dscrptorValueSscaled_train.txt'},\
			'Test':{'lig':'e:/brd4/PMF_Yolanda/dataSet/version3/testsetLigand/',\
					'pro':'e:/brd4/PMF_Yolanda/dataSet/version3/testsetPocket/',\
					'exp':'e:/brd4/PMF_Yolanda/dataSet/version3/dscrptorValueSscaled_test.txt'}
			}
	trainScaler = None
	dc = dist.UnformatedInput(dc_fn)
	pts = dist.PTS(dc)
	
	def tableGen(ligdir, prodir):
		dscrptorTable, activTable = [], []
		for num,p in enumerate(os.listdir(ligdir)):
			print p, num
			a = 0
			lig = pybel.readfile('sdf', ligdir+p).next()
			pro = pybel.readfile('pdb', prodir+p[:-4]+'.pdb').next()
			if lig.data['Active'] == '1': a = 1
			activTable.append([p[:-4].replace('ss','/'), str(a)])
			dd = GetDscrptor(lig, pro, pts)
			dscrptorTable.append([dd[d] for d in DESCRIPTOR])
		return dscrptorTable, activTable
	def tableOut(dscTable, acTable, expdir):
		with open(expdir, 'w') as f:
			f.write('\t'.join(['Name','Active']+DESCRIPTOR)+'\n')
			for ac,dsc in zip(dscTable, acTable):
				entry = ac + dsc
				f.write('\t'.join([str(i) for i in entry])+'\n')
	
	dscTabTrain, acTabTrain = tableGen(dirs['Train']['lig'], dirs['Train']['pro'])
	dscTabTrain_s, trainScaler = scaleDES(dscrptorTable, trainScaler)
	tableOut(dscTabTrain_s, acTabTrain, dirs['Train']['exp'])
	dscTabTest, acTabTest = tableGen(dirs['Test']['lig'], dirs['Test']['pro'])
	dscTabTest_s, trainScaler = scaleDES(dscrptorTable, trainScaler)
	tableOut(dscTabTest_s, acTabTest, dirs['Test']['exp'])
                
        


if __name__ == '__main__':
	#file_list = GetFileList('../Brd4_pdb_test1/doc.txt', '../Brd4_pdb_test1/struct/')
	#main('dist_scpdb_train_4.dat', '../Brd4_pdb_test1/testScore1.csv', file_list)
	dscrptorOut('dist_scPDB_train_4.dat')
	#dscrptors = singleTest('dist_scPDB_train_4.dat')
