from pybel import *
from random import shuffle

addData = '../addNewBrd4Inhibitors2015_docking.sdf'
ActiveNum, InactiveNum = 114, 18
ftrain = open('../addNewBrd4Inhibitors2015_trainset.sdf', 'w')
ftest = open('../addNewBrd4Inhibitors2015_testset.sdf', 'w')
testAct, testInact = range(ActiveNum), range(InactiveNum)
shuffle(testAct)
shuffle(testInact)
testAct, testInact = testAct[:ActiveNum/4], testInact[:InactiveNum/4]
idxAct, idxInact = 0, 0
for mol in readfile('sdf', addData):
    if mol.data['Active'] == '1':
        idxAct += 1
        if idxAct in testAct:
            ftest.write(mol.write('sdf'))
            print mol.title,'test'
        else:
            ftrain.write(mol.write('sdf'))
            print mol.title, 'train'
    elif mol.data['Active'] == '0':
        idxInact += 1
        if idxInact in testInact:
            ftest.write(mol.write('sdf'))
            print mol.title,'test'
        else:
            ftrain.write(mol.write('sdf'))
            print mol.title, 'train'


ftrain.close()
ftest.close()
