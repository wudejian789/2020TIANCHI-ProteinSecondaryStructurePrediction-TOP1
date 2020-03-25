import numpy as np
import torch,time,os,pickle
from torch import nn as nn
from nnLayer import *
from metrics import *
from collections import Counter,Iterable
from sklearn.model_selection import KFold

class BaseClassifier:
    def __init__(self):
        pass
    def calculate_y_logit(self, X, XLen):
        pass
    def cv_train(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1, augmentation=0.05,
                 optimType='Adam', lr=0.001, weightDecay=0, kFold=5, isHigherBetter=True, metrics="Score", report=["ACC", "MaF", "Score"], 
                 savePath='model'):
        kf = KFold(n_splits=kFold)
        validRes = []
        for i,(trainIndices,validIndices) in enumerate(kf.split(range(dataClass.totalSampleNum))):
            print(f'CV_{i+1}:')
            self.reset_parameters()
            dataClass.trainIdList,dataClass.validIdList = trainIndices,validIndices
            dataClass.trainSampleNum,self.validSampleNum = len(trainIndices),len(validIndices)
            dataClass.describe()
            res = self.train(dataClass,trainSize,batchSize,epoch,stopRounds,earlyStop,saveRounds,augmentation,optimType,lr,weightDecay,
                             isHigherBetter,metrics,report,f"{savePath}_cv{i+1}")
            validRes.append(res)
        Metrictor.table_show(validRes, report)
    def train(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1, augmentation=0.05,
              optimType='Adam', lr=0.001, weightDecay=0, isHigherBetter=True, metrics="Score", report=["ACC", "MaF", "Score"], 
              savePath='model'):
        assert batchSize%trainSize==0
        metrictor = Metrictor(dataClass.classNum)
        self.stepCounter = 0
        self.stepUpdate = batchSize//trainSize
        optimizer = torch.optim.Adam(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        trainStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='train', device=self.device, augmentation=augmentation)
        itersPerEpoch = (dataClass.trainSampleNum+trainSize-1)//trainSize
        mtc,bestMtc,stopSteps = 0.0,0.0,0
        if dataClass.validSampleNum>0: validStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='valid', device=self.device, augmentation=augmentation)
        st = time.time()
        for e in range(epoch):
            for i in range(itersPerEpoch):
                self.to_train_mode()
                X,Y = next(trainStream)
                loss = self._train_step(X,Y, optimizer)
                if stopRounds>0 and (e*itersPerEpoch+i+1)%stopRounds==0:
                    self.to_eval_mode()
                    print(f"After iters {e*itersPerEpoch+i+1}: [train] loss= {loss:.3f};", end='')
                    if dataClass.validSampleNum>0:
                        X,Y = next(validStream)
                        loss = self.calculate_loss(X,Y)
                        print(f' [valid] loss= {loss:.3f};', end='')
                    restNum = ((itersPerEpoch-i-1)+(epoch-e-1)*itersPerEpoch)*trainSize
                    speed = (e*itersPerEpoch+i+1)*trainSize/(time.time()-st)
                    print(" speed: %.3lf items/s; remaining time: %.3lfs;"%(speed, restNum/speed))
            if dataClass.validSampleNum>0 and (e+1)%saveRounds==0:
                self.to_eval_mode()
                print(f'========== Epoch:{e+1:5d} ==========')
                #Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
                #metrictor.set_data(Y_pre, Y)
                #print(f'[Total Train]',end='')
                #metrictor(report)
                print(f'[Total Valid]',end='')
                Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
                metrictor.set_data(Y_pre, Y)
                res = metrictor(report)
                mtc = res[metrics]
                print('=================================')
                if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                    print(f'Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!')
                    bestMtc = mtc
                    self.save("%s.pkl"%savePath, e+1, bestMtc, dataClass)
                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps>=earlyStop:
                        print(f'The val {metrics} has not improved for more than {earlyStop} steps in epoch {e+1}, stop training.')
                        break
        self.load("%s.pkl"%savePath)
        os.rename("%s.pkl"%savePath, "%s_%s.pkl"%(savePath, ("%.3lf"%bestMtc)[2:]))
        print(f'============ Result ============')
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
        metrictor.set_data(Y_pre, Y)
        print(f'[Total Train]',end='')
        metrictor(report)
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
        metrictor.set_data(Y_pre, Y)
        print(f'[Total Valid]',end='')
        res = metrictor(report)
        metrictor.each_class_indictor_show(dataClass.id2secItem)
        print(f'================================')
        return res
    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()
    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'] = dataClass.trainIdList,dataClass.validIdList
            stateDict['seqItem2id'],stateDict['id2seqItem'] = dataClass.seqItem2id,dataClass.id2seqItem
            stateDict['secItem2id'],stateDict['id2secItem'] = dataClass.secItem2id,dataClass.id2secItem
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            dataClass.trainIdList,dataClass.validIdList = parameters['trainIdList'],parameters['validIdList']
            dataClass.seqItem2id,dataClass.id2seqItem = parameters['seqItem2id'],parameters['id2seqItem']
            dataClass.secItem2id,dataClass.id2secItem = parameters['secItem2id'],parameters['id2secItem']
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)
        return torch.softmax(Y_pre, dim=1)
    def calculate_y(self, X):
        Y_pre = self.calculate_y_prob(X)
        return torch.argmax(Y_pre, dim=1)
    def calculate_loss(self, X, Y):
        Y_logit = self.calculate_y_logit(X)
        return self.criterion(Y_logit, Y)
    def calculate_indicator_by_iterator(self, dataStream, classNum, report):
        metrictor = Metrictor(classNum)
        Y_prob_pre,Y = self.calculate_y_prob_by_iterator(dataStream)
        metrictor.set_data(Y_prob_pre, Y)
        return metrictor(report)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.hstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr
    def calculate_y_by_iterator(self, dataStream):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        return Y_preArr.argmax(axis=1), YArr
    def to_train_mode(self):
        for module in self.moduleList:
            module.train()
    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()
    def _train_step(self, X, Y, optimizer):
        self.stepCounter += 1
        if self.stepCounter<self.stepUpdate:
            p = False
        else:
            self.stepCounter = 0
            p = True
        loss = self.calculate_loss(X, Y)/self.stepUpdate
        loss.backward()
        if p:
            optimizer.step()
            optimizer.zero_grad()
        return loss*self.stepUpdate

class TextClassifier_DeepCNN(BaseClassifier):
    def __init__(self, classNum, embedding, feaSize=128, filterNum=192, hiddenList=[], 
                 embDropout=0.3, fcDropout=0.5, 
                 useFocalLoss=False, weight=-1, device=torch.device("cuda:0")):
        self.textEmbedding = TextEmbedding( torch.tensor(embedding, dtype=torch.float),dropout=embDropout ).to(device)
        self.textCNN = TextDeepCNN(feaSize, filterNum).to(device)
        self.fcLinear = MLP(3*filterNum, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding, self.textCNN, self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight).to(device)
    def calculate_y_logit(self, X):
        X = self.textEmbedding(X['seqArr']) # => batchSizeh × seqLen × feaSize
        X = self.textCNN(X) # => batchSize × seqLen × scaleNum*filterNum
        return self.fcLinear(X) # => batchSize × seqLen × classNum
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)
        return torch.softmax(Y_pre, dim=2)
    def calculate_y(self, X):
        Y_pre = self.calculate_y_prob(X)
        return torch.argmax(Y_pre, dim=2)
    def calculate_y_by_iterator(self, dataStream):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        return Y_preArr.argmax(axis=2), YArr
    def calculate_loss(self, X, Y):
        Y_logit = self.calculate_y_logit(X)
        Y = Y.reshape(-1)
        Y_logit = Y_logit.reshape(len(Y),-1)
        return self.criterion(Y_logit, Y)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr


class TextClassifier_Transformer(BaseClassifier):
    def __init__(self, classNum, embedding, seqMaxLen, feaSize=128, dk=64, multiNum=8, hiddenList=[], 
                 embDropout=0.3, fcDropout=0.5, 
                 useFocalLoss=False, weight=-1, device=torch.device("cuda:0")):
        self.textEmbedding = TextEmbedding( torch.tensor(embedding, dtype=torch.float),dropout=embDropout ).to(device)
        self.textTransformer1 = TextTransformer(feaSize, dk, multiNum, seqMaxLen).to(device)
        self.fcLinear1 = LinearRelu(feaSize, feaSize*2, name='fcLinear1').to(device)
        self.textTransformer2 = TextTransformer(feaSize*2, dk, multiNum, seqMaxLen).to(device)
        self.fcLinear2 = LinearRelu(feaSize*2, feaSize*4, name='fcLinear2').to(device)
        self.textTransformer3 = TextTransformer(feaSize*4, dk, multiNum, seqMaxLen).to(device)
        self.fcLinear = MLP(feaSize*4, classNum, hiddenList, fcDropout).to(device)
        self.fcLinear1.name,self.fcLinear2.name = 'fcLinear1','fcLinear2'
        self.moduleList = nn.ModuleList([self.textEmbedding, 
                                         self.textTransformer1, self.textTransformer2, self.textTransformer3, 
                                         self.fcLinear1,self.fcLinear2,self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight).to(device)

        self.PE = torch.tensor([[[np.sin(pos/10000**(2*i/feaSize)) if i%2==0 else np.cos(pos/10000**(2*i/feaSize)) for i in range(1,feaSize+1)] for pos in range(1,seqMaxLen+1)]], dtype=torch.float32, device=device)
    def calculate_y_logit(self, X):
        X = self.textEmbedding(X['seqArr'])+self.PE # => batchSizeh × seqLen × feaSize
        X = self.textTransformer1(X) # => batchSize × seqLen × feaSize
        X = self.fcLinear1(X) # => batchSize × seqLen × feaSize*2
        X = self.textTransformer2(X) # => batchSize × seqLen × feaSize*2
        X = self.fcLinear2(X) # => batchSize × seqLen × feaSize*4
        X = self.textTransformer3(X) # => batchSize × seqLen × feaSize*4
        return self.fcLinear(X) # => batchSize × seqLen × classNum
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)
        return torch.softmax(Y_pre, dim=2)
    def calculate_y(self, X):
        Y_pre = self.calculate_y_prob(X)
        return torch.argmax(Y_pre, dim=2)
    def calculate_y_by_iterator(self, dataStream):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        return Y_preArr.argmax(axis=2), YArr
    def calculate_loss(self, X, Y):
        Y_logit = self.calculate_y_logit(X)
        Y = Y.reshape(-1)
        Y_logit = Y_logit.reshape(len(Y),-1)
        return self.criterion(Y_logit, Y)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr

class TextClassifier_BiGRU(BaseClassifier):
    def __init__(self, classNum, embedding, feaSize=128, hiddenSize=256, hiddenList=[], 
                 embDropout=0.3, fcDropout=0.5, num_layers=1,
                 useFocalLoss=False, weight=-1, device=torch.device("cuda:0")):
        self.textEmbedding = TextEmbedding( torch.tensor(embedding, dtype=torch.float),dropout=embDropout ).to(device)
        self.textBiGRU = TextBiGRU(feaSize, hiddenSize, num_layers=num_layers, dropout=0.1).to(device)
        self.fcLinear = MLP(hiddenSize*2, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding, self.textBiGRU, self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight).to(device)
    def calculate_y_logit(self, X):
        X,XLen = X['seqArr'],X['seqLenArr']
        X = self.textEmbedding(X)
        X = self.textBiGRU(X, None) # => batchSize × seqLen × hiddenSize*2
        return self.fcLinear(X)  # => batchSize × seqLen × classNum
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)
        return torch.softmax(Y_pre, dim=2)
    def calculate_y(self, X):
        Y_pre = self.calculate_y_prob(X)
        return torch.argmax(Y_pre, dim=2)
    def calculate_y_by_iterator(self, dataStream):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        return Y_preArr.argmax(axis=2), YArr
    def calculate_loss(self, X, Y):
        Y_logit = self.calculate_y_logit(X)
        Y = Y.reshape(-1)
        Y_logit = Y_logit.reshape(len(Y),-1)
        return self.criterion(Y_logit, Y)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr

class DCRNN(BaseClassifier):
    def __init__(self, classNum, embedding, feaSize=64, 
                 filterNum=64, contextSizeList=[3,7,11], 
                 hiddenSize=512, num_layers=3,
                 hiddenList=[2048], 
                 embDropout=0.2, BiGRUDropout=0.2, fcDropout=0.4, 
                 useFocalLoss=False, weight=-1, device=torch.device("cuda:0")):
        self.textEmbedding = TextEmbedding( torch.tensor(embedding, dtype=torch.float),dropout=embDropout ).to(device)
        self.textCNN = TextCNN( feaSize, contextSizeList, filterNum ).to(device)
        self.textBiGRU = TextBiGRU(len(contextSizeList)*filterNum, hiddenSize, num_layers=num_layers, dropout=BiGRUDropout).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterNum+hiddenSize*2, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding,self.textCNN,self.textBiGRU,self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight).to(device)
    def calculate_y_logit(self, X):
        X = X['seqArr']
        X = self.textEmbedding(X) # => batchSize × seqLen × feaSize
        X_conved = self.textCNN(X) # => batchSize × seqLen × scaleNum*filterNum
        X_BiGRUed = self.textBiGRU(X_conved, None) # => batchSize × seqLen × hiddenSize*2
        X = torch.cat([X_conved,X_BiGRUed], dim=2) # => batchSize × seqLen × (scaleNum*filterNum+hiddenSize*2)
        return self.fcLinear(X) # => batchSize × seqLen × classNum
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)
        return torch.softmax(Y_pre, dim=2)
    def calculate_y(self, X):
        Y_pre = self.calculate_y_prob(X)
        return torch.argmax(Y_pre, dim=2)
    def calculate_y_by_iterator(self, dataStream):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        return Y_preArr.argmax(axis=2), YArr
    def calculate_loss(self, X, Y):
        Y_logit = self.calculate_y_logit(X)
        Y = Y.reshape(-1)
        Y_logit = Y_logit.reshape(len(Y),-1)
        return self.criterion(Y_logit, Y)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr

class NormalNN(BaseClassifier):
    def __init__(self, classNum, embedding, seqMaxLen, feaSize, 
                 hiddenList=[2048], 
                 embDropout=0.2, fcDropout=0.4, 
                 useFocalLoss=False, weight=-1, device=torch.device("cuda:0")):
        self.textEmbedding = TextEmbedding( torch.tensor(embedding, dtype=torch.float),freeze=True,dropout=embDropout ).to(device)
        self.textContextNN = ContextNN( seqMaxLen ).to(device)
        self.fcLinear = MLP(feaSize*2, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding,self.textContextNN,self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight).to(device)
    def calculate_y_logit(self, X):
        X = X['seqArr']
        X = self.textEmbedding(X) # => batchSize × seqLen × feaSize
        context = self.textContextNN(X) # => batchSize × seqLen × feaSize
        X = torch.cat([X, context], dim=2) # => batchSize × seqLen × 2*feaSize
        return self.fcLinear(X) # => batchSize × seqLen × classNum
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)
        return torch.softmax(Y_pre, dim=2)
    def calculate_y(self, X):
        Y_pre = self.calculate_y_prob(X)
        return torch.argmax(Y_pre, dim=2)
    def calculate_y_by_iterator(self, dataStream):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        return Y_preArr.argmax(axis=2), YArr
    def calculate_loss(self, X, Y):
        Y_logit = self.calculate_y_logit(X)
        Y = Y.reshape(-1)
        Y_logit = Y_logit.reshape(len(Y),-1)
        return self.criterion(Y_logit, Y)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr

class OnehotNN(BaseClassifier):
    def __init__(self, classNum, embedding, seqLen, feaSize, 
                 hiddenList=[2048], 
                 embDropout=0.2, fcDropout=0.4, 
                 useFocalLoss=False, weight=-1, device=torch.device('cuda')):
        self.textEmbedding = TextEmbedding( torch.tensor(embedding, dtype=torch.float),freeze=True,dropout=embDropout ).to(device)
        self.fcLinear = MLP(feaSize*seqLen, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding, self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight).to(device)
    def calculate_y_logit(self, X):
        X = X['seqArr']
        X = self.textEmbedding(X) # => batchSize × seqLen × feaSize
        X = torch.flatten(X, start_dim=1) # => batchSize × seqLen*feaSize
        return self.fcLinear(X) # => batchSize × classNum


class FinalModel(BaseClassifier):
    def __init__(self, classNum, embedding, feaEmbedding, feaSize=64, 
                 filterNum=128, contextSizeList=[1,9,81], 
                 hiddenSize=512, num_layers=3,
                 hiddenList=[2048], 
                 embDropout=0.2, BiGRUDropout=0.2, fcDropout=0.4, 
                 useFocalLoss=False, weight=-1, device=torch.device("cuda:0")):
        self.textEmbedding = TextEmbedding( torch.tensor(embedding, dtype=torch.float),dropout=embDropout ).to(device)
        self.feaEmbedding = TextEmbedding( torch.tensor(feaEmbedding, dtype=torch.float),dropout=embDropout//2,name='feaEmbedding',freeze=True ).to(device)
        self.textCNN = TextCNN( feaSize, contextSizeList, filterNum ).to(device)
        self.textBiGRU = TextBiGRU(len(contextSizeList)*filterNum, hiddenSize, num_layers=num_layers, dropout=BiGRUDropout).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterNum+hiddenSize*2, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding,self.feaEmbedding,self.textCNN,self.textBiGRU,self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight).to(device)
    def calculate_y_logit(self, X):
        X = X['seqArr']
        X = torch.cat([self.textEmbedding(X),self.feaEmbedding(X)], dim=2) # => batchSize × seqLen × feaSize
        X_conved = self.textCNN(X) # => batchSize × seqLen × scaleNum*filterNum
        X_BiGRUed = self.textBiGRU(X_conved, None) # => batchSize × seqLen × hiddenSize*2
        X = torch.cat([X_conved,X_BiGRUed], dim=2) # => batchSize × seqLen × (scaleNum*filterNum+hiddenSize*2)
        return self.fcLinear(X) # => batchSize × seqLen × classNum
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)
        return torch.softmax(Y_pre, dim=2)
    def calculate_y(self, X):
        Y_pre = self.calculate_y_prob(X)
        return torch.argmax(Y_pre, dim=2)
    def calculate_y_by_iterator(self, dataStream):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        return Y_preArr.argmax(axis=2), YArr
    def calculate_loss(self, X, Y):
        Y_logit = self.calculate_y_logit(X)
        Y = Y.reshape(-1)
        Y_logit = Y_logit.reshape(len(Y),-1)
        return self.criterion(Y_logit, Y)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr