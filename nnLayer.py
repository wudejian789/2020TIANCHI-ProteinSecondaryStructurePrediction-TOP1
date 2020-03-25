from torch import nn as nn
from torch.nn import functional as F
import torch,time,os
import numpy as np

class TextEmbedding(nn.Module):
    def __init__(self, embedding, dropout=0.3, freeze=False, name='textEmbedding'):
        super(TextEmbedding, self).__init__()
        self.name = name
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=freeze)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        # x: batchSize × seqLen
        return self.dropout(self.embedding(x))

class TextCNN(nn.Module):
    def __init__(self, feaSize, contextSizeList, filterNum, name='textCNN'):
        super(TextCNN, self).__init__()
        self.name = name
        moduleList = []
        for i in range(len(contextSizeList)):
            moduleList.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=feaSize, out_channels=filterNum, kernel_size=contextSizeList[i], padding=contextSizeList[i]//2),
                    nn.BatchNorm1d(filterNum),
                    nn.ReLU(),
                    )
                )
        self.conv1dList = nn.ModuleList(moduleList)
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = x.transpose(1,2) # => batchSize × feaSize × seqLen
        x = [conv(x) for conv in self.conv1dList] # => scaleNum * (batchSize × filterNum × seqLen)
        return torch.cat(x, dim=1).transpose(1,2) # => batchSize × seqLen × scaleNum*filterNum

class TextDeepCNN(nn.Module):
    def __init__(self, feaSize, filterNum, name='textDeepCNN'):
        super(TextDeepCNN, self).__init__()
        self.name = name
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels=feaSize, out_channels=feaSize*2, kernel_size=1, padding=0),
                        nn.BatchNorm1d(feaSize*2),
                        nn.ReLU(),
                        nn.Conv1d(in_channels=feaSize*2, out_channels=filterNum, kernel_size=1, padding=0),
                        nn.BatchNorm1d(filterNum),
                        nn.ReLU(),
                        )
        self.conv2 = nn.Sequential(
                        nn.Conv1d(in_channels=feaSize, out_channels=feaSize*2, kernel_size=3, padding=1),
                        nn.BatchNorm1d(feaSize*2),
                        nn.ReLU(),
                        nn.Conv1d(in_channels=feaSize*2, out_channels=feaSize*2, kernel_size=3, padding=1),
                        nn.BatchNorm1d(feaSize*2),
                        nn.ReLU(),
                        nn.Conv1d(in_channels=feaSize*2, out_channels=filterNum, kernel_size=3, padding=1),
                        nn.BatchNorm1d(filterNum),
                        nn.ReLU(),
                        )
        self.conv3 = nn.Sequential(
                        nn.Conv1d(in_channels=feaSize, out_channels=feaSize*2, kernel_size=7, padding=3),
                        nn.BatchNorm1d(feaSize*2),
                        nn.ReLU(),
                        nn.Conv1d(in_channels=feaSize*2, out_channels=feaSize*4, kernel_size=3, padding=1),
                        nn.BatchNorm1d(feaSize*4),
                        nn.ReLU(),
                        nn.Conv1d(in_channels=feaSize*4, out_channels=feaSize*4, kernel_size=3, padding=1),
                        nn.BatchNorm1d(feaSize*4),
                        nn.ReLU(),
                        nn.Conv1d(in_channels=feaSize*4, out_channels=filterNum, kernel_size=3, padding=1),
                        nn.BatchNorm1d(filterNum),
                        nn.ReLU()
                        )
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = x.transpose(1,2) # => batchSize × feaSize × seqLen
        x = [self.conv1(x),self.conv2(x),self.conv3(x)] # => scaleNum * (batchSize × filterNum × seqLen)
        return torch.cat(x, dim=1).transpose(1,2) # => batchSize × seqLen × scaleNum*filterNum

class TextBiGRU(nn.Module):
    def __init__(self, feaSize, hiddenSize, num_layers=1, dropout=0.0, name='textBiGRU'):
        super(TextBiGRU, self).__init__()
        self.name = name
        self.biGRU = nn.GRU(feaSize, hiddenSize, bidirectional=True, batch_first=True, num_layers=num_layers, dropout=dropout)
    def forward(self, x, xlen=None):
        # x: batchSizeh × seqLen × feaSize
        if xlen is not None:
            xlen, indices = torch.sort(xlen, descending=True)
            _, desortedIndices = torch.sort(indices, descending=False)

            x = nn.utils.rnn.pack_padded_sequence(x[indices], xlen, batch_first=True)
        output, hn = self.biGRU(x) # output: batchSize × seqLen × hiddenSize*2; hn: numLayers*2 × batchSize × hiddenSize
        if xlen is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return output[desortedIndices]
        return output # output: batchSize × seqLen × hiddenSize*2

class TextTransformer(nn.Module):
    def __init__(self, featureSize, dk, multiNum, seqMaxLen, dropout=0.1, name='textTransformer'):
        super(TextTransformer, self).__init__()
        self.name = name
        self.dk = dk
        self.multiNum = multiNum
        self.WQ = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WK = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WV = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WO = nn.Linear(self.dk*multiNum, featureSize)
        self.layerNorm1 = nn.LayerNorm([seqMaxLen, featureSize])
        self.layerNorm2 = nn.LayerNorm([seqMaxLen, featureSize])
        self.Wffn = nn.Sequential(
                        nn.Linear(featureSize, featureSize*4), 
                        nn.ReLU(),
                        nn.Linear(featureSize*4, featureSize)
                    )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        queries = [self.WQ[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        keys = [self.WK[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        values = [self.WQ[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        score = [torch.bmm(queries[i], keys[i].transpose(1,2))/np.sqrt(self.dk) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × seqLen)
        z = [self.dropout(torch.bmm(F.softmax(score[i], dim=2), values[i])) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        z = self.WO(torch.cat(z, dim=2)) # => batch × seqLen × feaSize
        z = self.layerNorm1(x + z) # => batchSize × seqLen × feaSize
        ffnx = self.Wffn(z) # => batchSize × seqLen × feaSize
        return self.layerNorm2(z + ffnx) # => batchSize × seqLen × feaSize

class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gama=2, weight=-1, logit=True):
        super(FocalCrossEntropyLoss, self).__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float32), requires_grad=False)
        self.gama = gama
        self.logit = logit
    def forward(self, Y_pre, Y):
        if self.logit:
            Y_pre = F.softmax(Y_pre, dim=1)
        P = Y_pre[list(range(len(Y))), Y]
        if self.weight != -1:
            w = self.weight[Y]
        else:
            w = torch.tensor([1.0 for i in range(len(Y))], device=self.weight.device)
        w = (w/w.sum()).reshape(-1)
        return (-w*((1-P)**self.gama * torch.log(P))).sum()

class LinearRelu(nn.Module):
    def __init__(self, inSize, outSize, name='linearRelu'):
        super(LinearRelu, self).__init__()
        self.name = name
        self.layer = nn.Sequential(
                            nn.ReLU(),
                            nn.Linear(inSize, outSize),
                            nn.ReLU()
                         )
    def forward(self, x):
        return self.layer(x)

class MLP(nn.Module):
    def __init__(self, inSize, outSize, hiddenList=[], dropout=0.1, name='MLP', actFunc=nn.ReLU):
        super(MLP, self).__init__()
        self.name = name
        layers = nn.Sequential()
        for i,os in enumerate(hiddenList):
            layers.add_module(str(i*2), nn.Linear(inSize, os))
            layers.add_module(str(i*2+1), actFunc())
            inSize = os
        self.hiddenLayers = layers
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(inSize, outSize)
    def forward(self, x):
        x = self.hiddenLayers(x)
        return self.out(self.dropout(x))

class ContextNN(nn.Module):
    def __init__(self, seqMaxLen, name='contextNN'):
        super(ContextNN, self).__init__()
        self.name = name
        self.linear = nn.Linear(seqMaxLen, seqMaxLen)
        self.act = nn.ReLU()
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = x.transpose(1,2) # => batchSize × feaSize × seqLen
        x = self.linear(x).transpose(1,2) # => batchSize × seqLen × feaSize
        return self.act(x)