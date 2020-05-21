from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
import os,logging,pickle,random,torch
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

A_pssm = [0.872226004,0.35351513,0.291072682,0.312680229,0.340611722,0.221741296,0.186079503,0.158243969,0.323860653,0.284587638,0.245009849,0.335872147,0.414415237,0.3854859,0.462152589,0.415184604,0.415678055,0.162960674,0.301396213,0.20685564]
C_pssm = [0.131367262,0.993210741,0.051014628,0.070738663,0.141343773,0.100584273,0.083674474,0.252771442,0.099405786,0.13155555,0.076118584,0.084907803,0.09452313,0.097157337,0.099119889,0.088962404,0.12053233,0.263446631,0.118959045,0.039528232]
E_pssm = [0.244880490,0.296275879,0.647010458,0.882901417,0.374764691,0.099624506,0.069595497,0.203147438,0.418484161,0.264321695,0.066371629,0.662291457,0.444100017,0.393008897,0.439249980,0.373571161,0.322208045,0.082964380,0.166934354,0.131134815]
D_pssm = [0.296141334,0.249709671,0.850820407,0.512991913,0.265338090,0.166939991,0.110098353,0.170262548,0.507233108,0.157820406,0.104848505,0.431656491,0.494132735,0.374609189,0.448848640,0.442543182,0.263501172,0.173610625,0.197364754,0.224008087]
G_pssm = [0.134500843,0.138081132,0.083494184,0.077593987,0.092954086,0.914093711,0.289071932,0.239061378,0.087289090,0.205822566,0.465739703,0.147806816,0.154850238,0.122176565,0.150144272,0.147839920,0.183952228,0.602706325,0.333488820,0.761599454]
F_pssm = [0.342548214,0.270339354,0.195856200,0.171412621,0.922893893,0.103036555,0.088571455,0.214378914,0.223065930,0.095590198,0.063541687,0.275854077,0.281346349,0.229726456,0.385095037,0.233176818,0.209489150,0.096825142,0.123902125,0.076653113]
I_pssm = [0.176415172,0.188443798,0.308357998,0.275629523,0.235234856,0.260304213,0.082515323,0.932749953,0.306385104,0.140879156,0.107014411,0.360545061,0.366051301,0.258562316,0.334060816,0.326275232,0.232563130,0.174383626,0.204813114,0.460118278]
H_pssm = [0.143569076,0.134224827,0.104347336,0.128179912,0.109001577,0.199901391,0.908161712,0.144819942,0.130959143,0.426831691,0.605280251,0.196738287,0.138929626,0.151766316,0.136858734,0.137098916,0.232657396,0.400298615,0.673472124,0.211236731]
K_pssm = [0.298524389,0.266401812,0.568084594,0.301019405,0.265933301,0.195842778,0.121579844,0.212033239,0.861729735,0.150797931,0.154139928,0.407340393,0.416233547,0.350466473,0.424457415,0.672058897,0.319754166,0.293990392,0.199877922,0.308680273]
M_pssm = [0.173967682,0.197009826,0.081423949,0.105579666,0.094623978,0.336778656,0.659291616,0.172585245,0.174438823,0.524163387,0.886962625,0.179483812,0.216780932,0.145710941,0.134969375,0.170362862,0.206733556,0.533262289,0.384640005,0.244044113]
L_pssm = [0.225244504,0.376776187,0.138589819,0.183133318,0.135891614,0.295921134,0.491456844,0.175555301,0.226657379,0.866720079,0.660651867,0.275622096,0.322198802,0.142342904,0.229513750,0.280515594,0.271416295,0.449116352,0.328244811,0.313903458]
N_pssm = [0.269160755,0.292945305,0.336069790,0.480610819,0.406244664,0.119259286,0.077119964,0.359851389,0.465564987,0.193742239,0.102707801,0.890991538,0.404603890,0.356844609,0.474599142,0.378478471,0.359555731,0.113931650,0.148822188,0.178191469]
Q_pssm = [0.207780410,0.198091695,0.225152747,0.290446646,0.204574963,0.199898084,0.131305856,0.143066821,0.236687261,0.095375846,0.119815893,0.248063241,0.214212144,0.911767496,0.213092991,0.278199811,0.203782752,0.053982761,0.129363584,0.077917397]
P_pssm = [0.284015973,0.314122654,0.654988851,0.293965755,0.266342018,0.169070694,0.102684234,0.317582459,0.517885273,0.213771826,0.161182310,0.388242052,0.854016544,0.322383243,0.527796489,0.544931276,0.312645155,0.198176354,0.259373216,0.319545440]
S_pssm = [0.219297725,0.229428948,0.421884074,0.246614784,0.192544267,0.166972270,0.143466396,0.188121950,0.655457815,0.140421669,0.150345757,0.387298283,0.343485616,0.290221367,0.344720375,0.847964449,0.282228510,0.254715316,0.202529035,0.318086834]
R_pssm = [0.512894266,0.184875405,0.348063977,0.355832670,0.357881997,0.152903417,0.118138694,0.158965184,0.357595661,0.171367320,0.154680339,0.465237872,0.331957284,0.346368584,0.790428314,0.313318200,0.560004204,0.163479257,0.263011596,0.192724756]
T_pssm = [0.377764625,0.178184577,0.297722653,0.383766569,0.254015265,0.11150642,0.219367283,0.298850656,0.371877308,0.252202659,0.215385273,0.354912342,0.351367517,0.380481386,0.542486991,0.293440184,0.850656294,0.247789053,0.38727193,0.244643665]
W_pssm = [0.241772670,0.234871212,0.150171553,0.132149434,0.111593941,0.244663376,0.816923667,0.158818698,0.163135105,0.336656664,0.523681246,0.225822054,0.203272993,0.182922468,0.198275408,0.241880843,0.324984112,0.158786345,0.797769378,0.217122098]
V_pssm = [0.170517317,0.249649453,0.100918838,0.061540932,0.141034882,0.376889735,0.157983778,0.181397842,0.096812917,0.149128782,0.135011831,0.127537528,0.116118235,0.112229974,0.112107972,0.141627238,0.170548818,0.984635386,0.174473134,0.542319867]
Y_pssm = [0.500000000,0.119202922,0.268941421,0.268941421,0.268941421,0.268941421,0.268941421,0.268941421,0.268941421,0.268941421,0.268941421,0.268941421,0.268941421,0.119202922,0.500000000,0.268941421,0.500000000,0.119202922,0.268941421,0.268941421]
X_pssm = [0.203265404,0.327932961,0.196348043,0.162669050,0.179471469,0.730270333,0.182171579,0.397550899,0.153009421,0.190944510,0.321001024,0.269707976,0.198415836,0.253520609,0.200279215,0.259147101,0.267143332,0.775315115,0.313386476,0.987981675]
Z_pssm = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

featureDict = {
    'G':[57,   75,  156,   0.102,  0.085,  0.190,  0.152, 75.06714, 6.06, 2.34, 9.60, 48, 249.9, 0], # Glycine 甘氨酸
    'P':[57,   55,  152,   0.102,  0.301,  0.034,  0.068, 115.13194, 6.30, 1.99, 10.96, 90, 1620.0, 10.87], # Proline 脯氨酸
    'T':[83,  119,   96,   0.086,  0.108,  0.065,  0.079, 119.12034, 5.60, 2.09, 9.10, 93, 13.2, 1.67], # Threonine 苏氨酸
    'E':[151,   37,   74,   0.056,  0.060,  0.077,  0.064, 147.13074, 3.15, 2.10, 9.47, 109, 8.5, 2.09], # Glutamic Acid 谷氨酸
    'S':[77,   75,  143,   0.120,  0.139,  0.125,  0.106, 105.09344, 5.68, 2.21, 9.15, 73, 422.0, 1.25], # Serine 丝氨酸
    'K':[114,   74,  101,   0.055,  0.115,  0.072,  0.095, 146.18934, 9.60, 2.16, 9.06, 135, 739.0, 5.223888888888889], # Lysine 赖氨酸
    'C':[70,  119,  119,   0.149,  0.050,  0.117,  0.128, 121.15404, 5.05, 1.92, 10.70, 86, 280, 4.18], # Cysteine 半胱氨酸
    'L':[121,  130,   59,   0.061,  0.025,  0.036,  0.070, 131.17464, 6.01, 2.32, 9.58, 124, 21.7, 9.61], # Leucine 亮氨酸
    'M':[145,  105,   60,   0.068,  0.082,  0.014,  0.055, 149.20784, 5.74, 2.28, 9.21, 124, 56.2, 5.43], # Methionine 蛋氨酸
    'V':[106,  170,   50,   0.062,  0.048,  0.028,  0.053, 117.14784, 6.00, 2.29, 9.74, 105, 58.1, 6.27], # Valine 缬氨酸
    'D':[67,   89,  156,   0.161,  0.083,  0.191,  0.091, 133.10384, 2.85, 1.99, 9.90, 96, 5.0, 2.09], # Asparagine 天冬氨酸
    'A':[142,   83,   66,   0.06,   0.076,  0.035,  0.058, 89.09404, 6.01, 2.35, 9.87, 67, 167.2, 2.09], # Alanine 丙氨酸
    'R':[98,   93,   95,   0.070,  0.106,  0.099,  0.085, 174.20274, 10.76, 2.17, 9.04, 148, 855.6, 5.223888888888889], # Arginine 精氨酸
    'I':[108,  160,   47,   0.043,  0.034,  0.013,  0.056, 131.17464, 6.05, 2.32, 9.76, 124, 34.5, 12.54], # Isoleucine 异亮氨酸
    'N':[101,   54,  146,   0.147,  0.110,  0.179,  0.081, 132.11904, 5.41, 2.02, 8.80, 91, 28.5, 0], # Aspartic Acid 天冬酰胺
    'H':[100,   87,   95,   0.140,  0.047,  0.093,  0.054, 155.15634, 7.60, 1.80, 9.33, 118, 41.9, 2.09], # Histidine 组氨酸
    'F':[113,  138,   60,   0.059,  0.041,  0.065,  0.065, 165.19184, 5.49, 2.20, 9.60, 135, 27.6, 10.45], # Phenylalanine 苯丙氨酸
    'W':[108,  137,   96,   0.077,  0.013,  0.064,  0.167, 204.22844, 5.89, 2.46, 9.41, 163, 13.6, 14.21], # Tryptophan 色氨酸
    'Y':[69,  147,  114,   0.082,  0.065,  0.114,  0.125, 181.19124, 5.64, 2.20, 9.21, 141, 0.4, 9.61], # Tyrosine 酪氨酸
    'Q':[111,  110,   98,   0.074,  0.098,  0.037,  0.098, 146.14594, 5.65, 2.17, 9.13, 114, 4.7, -0.42], # Glutamine 谷氨酰胺
    'X':[99.9, 102.85, 99.15, 0.0887, 0.08429999999999999, 0.0824, 0.0875, 136.90127, 6.027, 2.1690000000000005, 0.0875, 109.2, 232.37999999999997, 5.223888888888889],
    'U':[99.9, 102.85, 99.15, 0.08870000000000001, 0.08430000000000001, 0.0824, 0.08750000000000001, 169.06, 6.026999999999999, 2.1690000000000005, 9.081309523809526, 109.19999999999999, 232.37999999999997, 5.223888888888889],
    'Z':[99.9, 102.85, 99.15, 0.08870000000000001, 0.08430000000000001, 0.0824, 0.08750000000000001, 136.90126999999998, 6.026999999999999, 2.1690000000000005, 9.081309523809526, 109.19999999999999, 232.37999999999997, 5.223888888888889],
}

class DataClass:
    def __init__(self, seqPath, secPath, validSize=0.3, k=3, minCount=10):
        # Open files and load data
        with open(seqPath,'r') as f:
            seqData = [' '*(k//2)+i[:-1]+' '*(k//2) for i in f.readlines()]
        with open(secPath,'r') as f:
            secData = [i[:-1] for i in f.readlines()]
        self.tmp,self.k = seqData,k
        seqData = [[seq[i-k//2:i+k//2+1] for i in range(k//2,len(seq)-k//2)] for seq in seqData]
        # Dropping uncommon items
        itemCounter = {}
        for seq in seqData:
            for i in seq:
                itemCounter[i] = itemCounter.get(i,0)+1
        seqData = [[i if itemCounter[i]>=minCount else "<UNK>" for i in seq] for seq in seqData]
        self.rawSeq,self.rawSec = seqData,secData
        self.minCount = minCount
        # Get mapping variables
        self.seqItem2id,self.id2seqItem = {"<EOS>":0, "<UNK>":1},["<EOS>", "<UNK>"]
        self.secItem2id,self.id2secItem = {"<EOS>":0},["<EOS>"]
        cnt = 2
        for seq in seqData:
            for i in seq:
                if i not in self.seqItem2id:
                    self.seqItem2id[i] = cnt
                    self.id2seqItem.append(i)
                    cnt += 1
        self.seqItemNum = cnt
        cnt = 1
        for sec in secData:
            for i in sec:
                if i not in self.secItem2id:
                    self.secItem2id[i] = cnt
                    self.id2secItem.append(i)
                    cnt += 1
        self.classNum = cnt
        # Tokenized the seq
        self.tokenizedSeq,self.tokenizedSec = np.array([[self.seqItem2id[i] for i in seq] for seq in seqData]),np.array([[self.secItem2id[i] for i in sec] for sec in secData])
        self.seqLen,self.secLen = np.array([len(seq)+1 for seq in seqData]),np.array([len(sec)+1 for sec in secData])
        self.trainIdList,self.validIdList = train_test_split(range(len(seqData)), test_size=validSize) if validSize>0.0 else (list(range(seqData)),[])
        self.trainSampleNum,self.validSampleNum = len(self.trainIdList),len(self.validIdList)
        self.totalSampleNum = self.trainSampleNum+self.validSampleNum
        self.vector = {}
        print('classNum:',self.classNum)
        print(f'seqItemNum:{self.seqItemNum}')
        print('train sample size:',len(self.trainIdList))
        print('valid sample size:',len(self.validIdList))
    def describe(self):
        pass
        '''
        trainSec,validSec = np.hstack(self.tokenizedSec[self.trainIdList]),np.hstack(self.tokenizedSec[self.validIdList])
        trainPad,validPad = self.trainSampleNum*self.seqLen.max()-len(trainSec),self.validSampleNum*self.seqLen.max()-len(validSec)
        trainSec,validSec = np.hstack([trainSec,[0]*trainPad]),np.hstack([validSec,[0]*validPad])
        print('===========DataClass Describe===========')
        print(f'{"CLASS":<16}{"TRAIN":<8}{"VALID":<8}')
        for i,c in enumerate(self.id2secItem):
            trainIsC = sum(trainSec==i)/self.trainSampleNum if self.trainSampleNum>0 else -1.0
            validIsC = sum(validSec==i)/self.validSampleNum if self.validSampleNum>0 else -1.0
            print(f'{c:<16}{trainIsC:<8.3f}{validIsC:<8.3f}')
        print('========================================')
        '''
    def vectorize(self, method="char2vec", feaSize=128, window=13, sg=1, 
                        workers=8, loadCache=True):
        if method=='feaEmbedding': loadCache = False
        vecPath = f'cache/{method}_k{self.k}_d{feaSize}.pkl'
        if os.path.exists(vecPath) and loadCache:
            with open(vecPath, 'rb') as f:
                self.vector['embedding'] = pickle.load(f)
            print(f'Loaded cache from cache/{vecPath}.')
            return
        if method == 'char2vec':
            doc = [list(i)+['<EOS>'] for i in self.rawSeq]
            model = Word2Vec(doc, min_count=self.minCount, window=window, size=feaSize, workers=workers, sg=sg, iter=10)
            char2vec = np.random.random((self.seqItemNum, feaSize))
            for i in range(self.seqItemNum):
                if self.id2seqItem[i] in model.wv:
                    char2vec[i] = model.wv[self.id2seqItem[i]]
                else:
                    print(self.id2seqItem[i],'not in training docs...')
            self.vector['embedding'] = char2vec
            with open(vecPath, 'wb') as f:
                pickle.dump(self.vector['embedding'], f, protocol=4)
        elif method == 'feaEmbedding':
            oh = np.eye(self.seqItemNum)
            feaAppend = []
            for i in range(self.seqItemNum):
                item = self.id2seqItem[i]
                if item in featureDict:
                    feaAppend.append( featureDict[item] )
                else:
                    feaAppend.append( np.random.random(14) )
            emb = np.hstack([oh, np.array(feaAppend)]).astype('float32')
            mean,std = emb.mean(axis=0),emb.std(axis=0)
            self.vector['feaEmbedding'] = (emb-mean)/(std+1e-10)

    def vector_merge(self, vecList, mergeVecName='mergeVec'):
        self.vector[mergeVec] = np.hstack([self.vector[i] for i in vecList])
        print(f'Get a new vector "{mergeVec}" with shape {self.vector[mergeVec].shape}...')

    def random_batch_data_stream(self, batchSize=128, type='train', device=torch.device('cpu'), augmentation=0.05):
        idList = [i for i in self.trainIdList] if type=='train' else [i for i in self.validIdList]
        X,XLen,Y = self.tokenizedSeq,self.seqLen,self.tokenizedSec
        seqMaxLen = XLen.max()
        while True:
            random.shuffle(idList)
            for i in range((len(idList)+batchSize-1)//batchSize):
                samples = idList[i*batchSize:(i+1)*batchSize]
                yield {
                        "seqArr":torch.tensor([[i if random.random()>augmentation else self.seqItem2id['<UNK>'] for i in seq]+[0]*(seqMaxLen-len(seq)) for seq in X[samples]], dtype=torch.long).to(device), \
                        "seqLenArr":torch.tensor(XLen[samples], dtype=torch.int).to(device)
                      }, torch.tensor([i+[0]*(seqMaxLen-len(i)) for i in Y[samples]], dtype=torch.long).to(device)
    
    def one_epoch_batch_data_stream(self, batchSize=128, type='valid', device=torch.device('cpu')):
        idList = [i for i in self.trainIdList] if type=='train' else [i for i in self.validIdList]
        X,XLen,Y = self.tokenizedSeq,self.seqLen,self.tokenizedSec
        seqMaxLen = XLen.max()
        for i in range((len(idList)+batchSize-1)//batchSize):
            samples = idList[i*batchSize:(i+1)*batchSize]
            yield {
                    "seqArr":torch.tensor([i+[0]*(seqMaxLen-len(i)) for i in X[samples]], dtype=torch.long).to(device), \
                    "seqLenArr":torch.tensor(XLen[samples], dtype=torch.int).to(device)
                  }, torch.tensor([i+[0]*(seqMaxLen-len(i)) for i in Y[samples]], dtype=torch.long).to(device)

'''
P(α):形成α螺旋的可能性 
P(β):形成β螺旋的可能性
P(turn):转向概率
f(i):
f(i+1):
f(i+2):
f(i+3):
分子量(Da):
pl:等电点
pK1(α-COOH):解离常数
pK2(α-NH3):解离常数
范德华半径:
水中溶解度(25℃,g/L):
氨基侧链疏水性(乙醇->水,kj/mol)
'''

class DataClass_BP:
    def __init__(self, seqPath, secPath, validSize=0.3):
        # Open files and load data
        with open(seqPath,'r') as f:
            seqData = [i[:-1].replace('U','X').replace('Z','X') for i in f.readlines()]
        with open(secPath,'r') as f:
            secData = [i[:-1] for i in f.readlines()]
        self.rawSeq,self.rawSec = seqData,secData
        # Get mapping variables
        self.seqItem2id,self.id2seqItem = {"<EOS>":0, "<UNK>":1},["<EOS>", "<UNK>"]
        self.secItem2id,self.id2secItem = {"<EOS>":0},["<EOS>"]
        cnt = 2
        for seq in seqData:
            for i in seq:
                if i not in self.seqItem2id:
                    self.seqItem2id[i] = cnt
                    self.id2seqItem.append(i)
                    cnt += 1
        self.seqItemNum = cnt
        cnt = 1
        for sec in secData:
            for i in sec:
                if i not in self.secItem2id:
                    self.secItem2id[i] = cnt
                    self.id2secItem.append(i)
                    cnt += 1
        self.classNum = cnt
        # Tokenized the seq
        self.seqLen,self.secLen = np.array([len(seq)+1 for seq in seqData]),np.array([len(sec)+1 for sec in secData])
        self.seqMaxLen = np.max(self.seqLen)
        self.tokenizedSeq,self.tokenizedSec = np.array([[self.seqItem2id[i] for i in seq]+[self.seqItem2id['<EOS>']]*(self.seqMaxLen-len(seq)) for seq in seqData]),np.array([[self.secItem2id[i] for i in sec]+[self.secItem2id['<EOS>']]*(self.seqMaxLen-len(sec)) for sec in secData])
        self.trainIdList,self.validIdList = train_test_split(range(len(seqData)), test_size=validSize) if validSize>0.0 else (list(range(seqData)),[])
        self.trainSampleNum,self.validSampleNum = len(self.trainIdList),len(self.validIdList)
        self.totalSampleNum = self.trainSampleNum+self.validSampleNum
        self.vector = {}
        print('classNum:',self.classNum)
        print(f'seqItemNum:{self.seqItemNum}')
        print('train sample size:',len(self.trainIdList))
        print('valid sample size:',len(self.validIdList))
    def describe(self):
        trainSec,validSec = np.hstack(self.tokenizedSec[self.trainIdList]),np.hstack(self.tokenizedSec[self.validIdList])
        trainPad,validPad = self.trainSampleNum*self.seqLen.max()-len(trainSec),self.validSampleNum*self.seqLen.max()-len(validSec)
        trainSec,validSec = np.hstack([trainSec,[0]*trainPad]),np.hstack([validSec,[0]*validPad])
        print('===========DataClass Describe===========')
        print(f'{"CLASS":<16}{"TRAIN":<8}{"VALID":<8}')
        for i,c in enumerate(self.id2secItem):
            trainIsC = sum(trainSec==i)/self.trainSampleNum if self.trainSampleNum>0 else -1.0
            validIsC = sum(validSec==i)/self.validSampleNum if self.validSampleNum>0 else -1.0
            print(f'{c:<16}{trainIsC:<8.3f}{validIsC:<8.3f}')
        print('========================================')
    def vectorize(self, method="feaEmbedding", 
                  feaSize=128, window=13, sg=1, 
                  workers=8, loadCache=True):
        if method == 'feaEmbedding':
            vecPath = f'cache/{method}.pkl'
        else:
            vecPath = f'cache/{method}_k{self.k}_d{feaSize}.pkl'
        if os.path.exists(vecPath) and loadCache:
            with open(vecPath, 'rb') as f:
                self.vector['embedding'] = pickle.load(f)
            print(f'Loaded cache from cache/{vecPath}.')
            return
        if method == 'feaEmbedding':
            oh = np.eye(self.seqItemNum)
            feaAppend = []
            for i in range(self.seqItemNum):
                item = self.id2seqItem[i]
                if item in featureDict:
                    feaAppend.append( featureDict[item] )
                else:
                    feaAppend.append( np.random.random(14) )
            emb = np.hstack([oh, np.array(feaAppend)]).astype('float32')
            mean,std = emb.mean(axis=0),emb.std(axis=0)
            self.vector['embedding'] = (emb-mean)/(std+1e-10)
        elif method == 'char2vec':
            doc = [list(i)+['<EOS>'] for i in self.rawSeq]
            model = Word2Vec(doc, min_count=self.minCount, window=window, size=feaSize, workers=workers, sg=sg, iter=10)
            char2vec = np.random.random((self.seqItemNum, feaSize))
            for i in range(self.seqItemNum):
                if self.id2seqItem[i] in model.wv:
                    char2vec[i] = model.wv[self.id2seqItem[i]]
                else:
                    print(self.id2seqItem[i],'not in training docs...')
            self.vector['embedding'] = char2vec

        with open(vecPath, 'wb') as f:
            pickle.dump(self.vector['embedding'], f, protocol=4)

    def vector_merge(self, vecList, mergeVecName='mergeVec'):
        self.vector[mergeVec] = np.hstack([self.vector[i] for i in vecList])
        print(f'Get a new vector "{mergeVec}" with shape {self.vector[mergeVec].shape}...')

    def random_batch_data_stream(self, batchSize=128, type='train', device=torch.device('cpu'), augmentation=0.05):
        idList = [i for i in self.trainIdList] if type=='train' else [i for i in self.validIdList]
        X,XLen,Y = self.tokenizedSeq,self.seqLen,self.tokenizedSec
        while True:
            random.shuffle(idList)
            for i in range((len(idList)+batchSize-1)//batchSize):
                samples = idList[i*batchSize:(i+1)*batchSize]
                yield {
                        "seqArr":torch.tensor([[i if random.random()>augmentation else self.seqItem2id['<UNK>'] for i in seq] for seq in X[samples]], dtype=torch.long).to(device), \
                        "seqLenArr":torch.tensor(XLen[samples], dtype=torch.int).to(device)
                      }, torch.tensor([i for i in Y[samples]], dtype=torch.long).to(device)
    
    def one_epoch_batch_data_stream(self, batchSize=128, type='valid', device=torch.device('cpu')):
        idList = [i for i in self.trainIdList] if type=='train' else [i for i in self.validIdList]
        X,XLen,Y = self.tokenizedSeq,self.seqLen,self.tokenizedSec
        for i in range((len(idList)+batchSize-1)//batchSize):
            samples = idList[i*batchSize:(i+1)*batchSize]
            yield {
                    "seqArr":torch.tensor([i for i in X[samples]], dtype=torch.long).to(device), \
                    "seqLenArr":torch.tensor(XLen[samples], dtype=torch.int).to(device)
                  }, torch.tensor([i for i in Y[samples]], dtype=torch.long).to(device)

class DataClass_BP2:
    def __init__(self, seqPath, secPath, window=17, validSize=0.3):
        # Open files and load data
        with open(seqPath,'r') as f:
            seqData = []
            for seq in f.readlines():
                seq = ' '*(window//2) + seq[:-1] + ' '*(window//2)
                seqData += [seq[i-window//2:i+window//2+1] for i in range(window//2,len(seq)-window//2)]
        with open(secPath,'r') as f:
            secData = []
            for sec in f.readlines():
                secData += list(sec[:-1])
        self.seqLen = window
        self.rawSeq,self.rawSec = seqData,secData
        # Get mapping variables
        self.seqItem2id,self.id2seqItem = {"<UNK>":0},["<UNK>"]
        self.secItem2id,self.id2secItem = {},[]
        cnt = 1
        for seq in seqData:
            for i in seq:
                if i not in self.seqItem2id:
                    self.seqItem2id[i] = cnt
                    self.id2seqItem.append(i)
                    cnt += 1
        self.seqItemNum = cnt
        cnt = 0
        for sec in secData:
            for i in sec:
                if i not in self.secItem2id:
                    self.secItem2id[i] = cnt
                    self.id2secItem.append(i)
                    cnt += 1
        self.classNum = cnt
        # Tokenized the seq
        self.tokenizedSeq,self.tokenizedSec = np.array([[self.seqItem2id[i] for i in seq] for seq in seqData]),np.array([self.secItem2id[sec] for sec in secData])
        self.trainIdList,self.validIdList = train_test_split(range(len(seqData)), test_size=validSize, stratify=self.tokenizedSec) if validSize>0.0 else (list(range(seqData)),[])
        self.trainSampleNum,self.validSampleNum = len(self.trainIdList),len(self.validIdList)
        self.totalSampleNum = self.trainSampleNum+self.validSampleNum
        self.vector = {}
        print('classNum:',self.classNum)
        print(f'seqItemNum:{self.seqItemNum}')
        print('train sample size:',len(self.trainIdList))
        print('valid sample size:',len(self.validIdList))
    def describe(self):
        trainSec,validSec = np.hstack(self.tokenizedSec[self.trainIdList]),np.hstack(self.tokenizedSec[self.validIdList])
        trainPad,validPad = self.trainSampleNum*self.seqLen.max()-len(trainSec),self.validSampleNum*self.seqLen.max()-len(validSec)
        trainSec,validSec = np.hstack([trainSec,[0]*trainPad]),np.hstack([validSec,[0]*validPad])
        print('===========DataClass Describe===========')
        print(f'{"CLASS":<16}{"TRAIN":<8}{"VALID":<8}')
        for i,c in enumerate(self.id2secItem):
            trainIsC = sum(trainSec==i)/self.trainSampleNum if self.trainSampleNum>0 else -1.0
            validIsC = sum(validSec==i)/self.validSampleNum if self.validSampleNum>0 else -1.0
            print(f'{c:<16}{trainIsC:<8.3f}{validIsC:<8.3f}')
        print('========================================')
    def vectorize(self, method="feaEmbedding", loadCache=True):
        vecPath = f'cache/{method}.pkl'
        if os.path.exists(vecPath) and loadCache:
            with open(vecPath, 'rb') as f:
                self.vector['embedding'] = pickle.load(f)
            print(f'Loaded cache from cache/{vecPath}.')
            return
        if method == 'feaEmbedding':
            oh = np.eye(self.seqItemNum)
            feaAppend = []
            for i in range(self.seqItemNum):
                item = self.id2seqItem[i]
                if item in featureDict:
                    feaAppend.append( featureDict[item] )
                else:
                    feaAppend.append( np.random.random(14) )
            emb = np.hstack([oh, np.array(feaAppend)]).astype('float32')
            mean,std = emb.mean(axis=0),emb.std(axis=0)
            self.vector['embedding'] = (emb-mean)/(std+1e-10)
            self.vector['embedding'][self.seqItem2id[' ']] *= 0
        with open(vecPath, 'wb') as f:
            pickle.dump(self.vector['embedding'], f, protocol=4)

    def vector_merge(self, vecList, mergeVecName='mergeVec'):
        self.vector[mergeVec] = np.hstack([self.vector[i] for i in vecList])
        print(f'Get a new vector "{mergeVec}" with shape {self.vector[mergeVec].shape}...')

    def random_batch_data_stream(self, batchSize=128, type='train', device=torch.device('cpu'), augmentation=0.05):
        idList = [i for i in self.trainIdList] if type=='train' else [i for i in self.validIdList]
        X,Y = self.tokenizedSeq,self.tokenizedSec
        while True:
            random.shuffle(idList)
            for i in range((len(idList)+batchSize-1)//batchSize):
                samples = idList[i*batchSize:(i+1)*batchSize]
                yield {
                        "seqArr":torch.tensor([[i if random.random()>augmentation else self.seqItem2id['<UNK>'] for i in seq] for seq in X[samples]], dtype=torch.long).to(device), \
                      }, torch.tensor([i for i in Y[samples]], dtype=torch.long).to(device)
    
    def one_epoch_batch_data_stream(self, batchSize=128, type='valid', device=torch.device('cpu')):
        idList = [i for i in self.trainIdList] if type=='train' else [i for i in self.validIdList]
        X,Y = self.tokenizedSeq,self.tokenizedSec
        for i in range((len(idList)+batchSize-1)//batchSize):
            samples = idList[i*batchSize:(i+1)*batchSize]
            yield {
                    "seqArr":torch.tensor([i for i in X[samples]], dtype=torch.long).to(device), \
                  }, torch.tensor([i for i in Y[samples]], dtype=torch.long).to(device)