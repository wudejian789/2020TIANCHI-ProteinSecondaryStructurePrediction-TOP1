from nnLayer import *
from torch.nn import functional as F
from tqdm import tqdm
import pickle

class Predictor_DCRNN:
    def __init__(self, weightPath, classNum=9, 
                 feaSize=128, filterNum=64, contextSizeList=[3,7,11], 
                 hiddenSize=512, num_layers=3,
                 hiddenList=[2048], 
                 map_location="cpu", device=torch.device("cpu")):
        stateDict = torch.load(weightPath, map_location=map_location)
        self.seqItem2id,self.id2seqItem = stateDict['seqItem2id'],stateDict['id2seqItem']
        self.secItem2id,self.id2secItem = stateDict['secItem2id'],stateDict['id2secItem']
        self.k = int(weightPath[:-4].split('_')[-2][1:])
        self.trainIdList,self.validIdList = stateDict['trainIdList'],stateDict['validIdList']
        self.seqItem2id,self.id2seqItem = stateDict['seqItem2id'],stateDict['id2seqItem']
        self.secItem2id,self.id2secItem = stateDict['secItem2id'],stateDict['id2secItem']
        self.textEmbedding = TextEmbedding( torch.zeros((len(self.id2seqItem),feaSize), dtype=torch.float32) ).to(device)
        self.textCNN = TextCNN( feaSize, contextSizeList, filterNum ).to(device)
        self.textBiGRU = TextBiGRU(len(contextSizeList)*filterNum, hiddenSize, num_layers=num_layers).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterNum+hiddenSize*2, classNum, hiddenList).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding,self.textCNN,self.textBiGRU,self.fcLinear])
        for module in self.moduleList:
            module.load_state_dict(stateDict[module.name])
            module.eval()
        self.device = device
        print("%d epochs and %.3lf val Score 's model load finished."%(stateDict['epochs'], stateDict['bestMtc']))

    def predict(self, seqData, batchSize=32):
        if type(seqData)==str:
            with open(seqData, 'r') as f:
                seqData = f.readlines()
        k = self.k
        seqData = [' '*(k//2)+i[:-1]+' '*(k//2) for i in seqData]
        seqData = [[seq[i-k//2:i+k//2+1] for i in range(k//2,len(seq)-k//2)] for seq in seqData]
        tokenizedSeq = np.array([[self.seqItem2id[i] if i in self.seqItem2id else self.seqItem2id['<UNK>'] for i in seq] for seq in seqData])
        seqMaxLen = np.array([len(seq)+1 for seq in seqData]).max()

        secPre = []
        idList = list(range(len(tokenizedSeq)))
        print('Predicting...')
        for i in tqdm(range((len(idList)+batchSize-1)//batchSize)):
            samples = idList[i*batchSize:(i+1)*batchSize]
            batchSeq = torch.tensor([i+[0]*(seqMaxLen-len(i)) for i in tokenizedSeq[samples]], dtype=torch.long).to(self.device)
            batchSec = F.softmax(self._calculate_y_logit(batchSeq), dim=2).cpu().data.numpy()
            secPre.append(batchSec)
        secPre = np.vstack(secPre).astype('float32')
        print('Finished!')
        return secPre,[len(seq) for seq in seqData]

    def _calculate_y_logit(self, X):
        X = self.textEmbedding(X) # => batchSize × seqLen × feaSize
        X_conved = self.textCNN(X) # => batchSize × seqLen × scaleNum*filterNum
        X_BiGRUed = self.textBiGRU(X_conved, None) # => batchSize × seqLen × hiddenSize*2
        X = torch.cat([X_conved,X_BiGRUed], dim=2) # => batchSize × seqLen × (scaleNum*filterNum+hiddenSize*2)
        return self.fcLinear(X) # => batchSize × seqLen × classNum

class Predictor_OneHotBP:
    def __init__(self, weightPath, classNum=8, 
                 feaSize=39, seqLen=79, hiddenList=[2048], 
                 map_location="cpu", device=torch.device("cpu")):
        stateDict = torch.load(weightPath, map_location=map_location)
        self.seqItem2id,self.id2seqItem = stateDict['seqItem2id'],stateDict['id2seqItem']
        self.secItem2id,self.id2secItem = stateDict['secItem2id'],stateDict['id2secItem']
        self.window = seqLen
        self.trainIdList,self.validIdList = stateDict['trainIdList'],stateDict['validIdList']
        self.seqItem2id,self.id2seqItem = stateDict['seqItem2id'],stateDict['id2seqItem']
        self.secItem2id,self.id2secItem = stateDict['secItem2id'],stateDict['id2secItem']
        self.textEmbedding = TextEmbedding( torch.zeros((len(self.id2seqItem),feaSize), dtype=torch.float32) ).to(device)
        self.fcLinear = MLP(feaSize*seqLen, classNum, hiddenList).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding,self.fcLinear])
        for module in self.moduleList:
            module.load_state_dict(stateDict[module.name])
            module.eval()
        self.device = device
        print("%d epochs and %.3lf val Score 's model load finished."%(stateDict['epochs'], stateDict['bestMtc']))

    def predict(self, seqData, batchSize=10240):
        window = self.window
        if type(seqData)==str:
            with open(seqData, 'r') as f:
                rawData = [i[:-1] for i in f.readlines()]

        seqData = []
        for seq in rawData:
            seq = ' '*(window//2) + seq + ' '*(window//2)
            seqData += [seq[i-window//2:i+window//2+1] for i in range(window//2,len(seq)-window//2)]
        tokenizedSeq = np.array([[self.seqItem2id[i] if i in self.seqItem2id else self.seqItem2id['<UNK>'] for i in seq] for seq in seqData])

        secPre = []
        idList = list(range(len(tokenizedSeq)))
        print('Predicting...')
        for i in tqdm(range((len(idList)+batchSize-1)//batchSize)):
            samples = idList[i*batchSize:(i+1)*batchSize]
            batchSeq = torch.tensor(tokenizedSeq[samples], dtype=torch.long).to(self.device)
            batchSec = F.softmax(self._calculate_y_logit(batchSeq), dim=1).cpu().data.numpy()
            secPre.append(batchSec)
        secPre = np.vstack(secPre).astype('float32')
        print('Finished!')
        return secPre,[len(seq) for seq in rawData]

    def _calculate_y_logit(self, X):
        X = self.textEmbedding(X) # => batchSize × seqLen × feaSize
        X = torch.flatten(X, start_dim=1) # => batchSize × seqLen*feaSize
        return self.fcLinear(X) # => batchSize × classNum

class Predictor_final:
    def __init__(self, weightPath, classNum=9, 
                 feaSize=64, filterNum=128, contextSizeList=[1,9,81], 
                 hiddenSize=512, num_layers=3,
                 hiddenList=[2048], 
                 map_location="cpu", device=torch.device("cpu")):
        stateDict = torch.load(weightPath, map_location=map_location)
        self.seqItem2id,self.id2seqItem = stateDict['seqItem2id'],stateDict['id2seqItem']
        self.secItem2id,self.id2secItem = stateDict['secItem2id'],stateDict['id2secItem']
        self.trainIdList,self.validIdList = stateDict['trainIdList'],stateDict['validIdList']
        self.seqItem2id,self.id2seqItem = stateDict['seqItem2id'],stateDict['id2seqItem']
        self.secItem2id,self.id2secItem = stateDict['secItem2id'],stateDict['id2secItem']
        self.textEmbedding = TextEmbedding( torch.zeros((len(self.id2seqItem),feaSize-39), dtype=torch.float32) ).to(device)
        self.feaEmbedding = TextEmbedding( torch.zeros((len(self.id2seqItem),39), dtype=torch.float32), freeze=True, name='feaEmbedding' ).to(device)
        self.textCNN = TextCNN( feaSize, contextSizeList, filterNum ).to(device)
        self.textBiGRU = TextBiGRU(len(contextSizeList)*filterNum, hiddenSize, num_layers=num_layers).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterNum+hiddenSize*2, classNum, hiddenList).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding,self.feaEmbedding,self.textCNN,self.textBiGRU,self.fcLinear])
        for module in self.moduleList:
            module.load_state_dict(stateDict[module.name])
            module.eval()
        self.device = device
        print("%d epochs and %.3lf val Score 's model load finished."%(stateDict['epochs'], stateDict['bestMtc']))

    def predict(self, seqData, batchSize=32):
        if type(seqData)==str:
            with open(seqData, 'r') as f:
                seqData = f.readlines()
        seqData = [i[:-1] for i in seqData]
        tokenizedSeq = np.array([[self.seqItem2id[i] if i in self.seqItem2id else self.seqItem2id['<UNK>'] for i in seq] for seq in seqData])
        seqMaxLen = np.array([len(seq)+1 for seq in seqData]).max()

        secPre = []
        idList = list(range(len(tokenizedSeq)))
        print('Predicting...')
        for i in tqdm(range((len(idList)+batchSize-1)//batchSize)):
            samples = idList[i*batchSize:(i+1)*batchSize]
            batchSeq = torch.tensor([i+[0]*(seqMaxLen-len(i)) for i in tokenizedSeq[samples]], dtype=torch.long).to(self.device)
            batchSec = F.softmax(self._calculate_y_logit(batchSeq), dim=2).cpu().data.numpy()
            secPre.append(batchSec)
        secPre = np.vstack(secPre).astype('float32')
        print('Finished!')
        return secPre,[len(seq) for seq in seqData]

    def _calculate_y_logit(self, X):
        X = torch.cat([self.textEmbedding(X),self.feaEmbedding(X)], dim=2) # => batchSize × seqLen × feaSize
        X_conved = self.textCNN(X) # => batchSize × seqLen × scaleNum*filterNum
        X_BiGRUed = self.textBiGRU(X_conved, None) # => batchSize × seqLen × hiddenSize*2
        X = torch.cat([X_conved,X_BiGRUed], dim=2) # => batchSize × seqLen × (scaleNum*filterNum+hiddenSize*2)
        return self.fcLinear(X) # => batchSize × seqLen × classNum