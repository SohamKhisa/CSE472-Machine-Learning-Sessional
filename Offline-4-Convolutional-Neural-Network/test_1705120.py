import os
import cv2
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import sys
import pickle



'''Convolution Layer'''
class Convolution:
    def __init__(self, noutchannel, filterdim, stride, padding):
        self.noutchannel = noutchannel
        self.filterdim = filterdim
        self.stride = stride
        self.padding = padding
        self.isfirst = True
    
    def forward(self, input):
        self.input = input
        nsamples, inputh, inputw, indepth = input.shape
        if self.isfirst == True:
            self.filters = np.random.randn(self.filterdim, self.filterdim, indepth, self.noutchannel) * np.sqrt(2 /(self.filterdim * self.filterdim * indepth))
            self.biases = np.zeros(self.noutchannel)
            self.isfirst = False
        outputh = int((inputh - self.filterdim + 2 * self.padding) / self.stride + 1)
        outputw = int((inputw - self.filterdim + 2 * self.padding) / self.stride + 1)
        output = np.zeros((nsamples, outputh, outputw, self.noutchannel))

        self.input = np.pad(input, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)), 'constant')
        # do forward convolution for 
        for m in range(nsamples):
            for i in range(outputh):
                for j in range(outputw):
                    for k in range(self.noutchannel):
                        output[m, i, j, k] = np.sum(self.input[m, i * self.stride : i * self.stride + self.filterdim, j * self.stride : j * self.stride + self.filterdim, :] * self.filters[:, :, :, k]) + self.biases[k]
        
        # self.input = self.input[:, self.padding:-self.padding, self.padding:-self.padding, :]
        return output
    
    def backward(self, dz, learningrate):
        dLfilter = np.zeros(self.filters.shape)
        # output = self.input.astype(np.float64)
        output = np.zeros(self.input.shape)
        # print(dLout.shape)
        nsamples = dz.shape[0]
        tempfilter = np.rot90(self.filters, 2, axes=(0, 1))
        dB = np.sum(dz, axis=(0,1,2))/nsamples

        # print(tempfilter.shape, self.filters.shape, dz.shape)
        for m in range (nsamples):
            for i in range(dz.shape[1]):
                for j in range(dz.shape[2]):
                    for k in range(dz.shape[3]):
                        dLfilter[:, :, :, k] += dz[m, i, j, k] * self.input[m, i * self.stride : i * self.stride + self.filterdim, j * self.stride : j * self.stride + self.filterdim, :]
                        output[m, i * self.stride : i * self.stride + self.filterdim, j * self.stride : j * self.stride + self.filterdim, :] += tempfilter[:, :, :, k] * dz[m, i, j, k]
        # clip the gradient
        dLfilter = np.clip(dLfilter, -1, 1)
        dB = np.clip(dB, -1, 1)
        self.filters -= learningrate*dLfilter
        self.biases -= learningrate*dB
        output = output[:, self.padding:-self.padding, self.padding:-self.padding, :]
        return output
    
    def getWeights(self):
        return self.filters, self.biases
    
    def setWeights(self, filters, biases):
        self.filters = filters
        self.biases = biases
    
    def weightExist(self):
        return True


'''ReLU'''
class Relu:
    def __init__(self):
        pass
    
    def forward(self, input):
        return np.maximum(0, input)

    def backward(self, dLout, learningrate):
        out = (dLout > 0) * dLout
        # print(out.shape)
        return out

    def getWeights(self):
        return None

    def setWeights(self, weights, biases):
        pass

    def weightExist(self):
        return False




'''Max Pooling'''
class Maxpooling:
    def __init__(self, filterdim, stride):
        self.filterdim = filterdim
        self.stride = stride
    
    def forward(self, input):
        self.input = input
        nsamples, inputh, inputw, indepth = input.shape
        outputh = int((inputh - self.filterdim) / self.stride + 1)
        outputw = int((inputw - self.filterdim) / self.stride + 1)
        output = np.zeros((nsamples, outputh, outputw, indepth))
        for m in range(nsamples):
            for i in range(outputh):
                for j in range(outputw):
                    for k in range(indepth):
                        output[m, i, j, k] = np.max(input[m, i * self.stride : i * self.stride + self.filterdim, j * self.stride : j * self.stride + self.filterdim, k])
        return output
    
    def backward(self, dLout, learningrate):
        dLin = np.zeros(self.input.shape)
        nsamples = dLout.shape[0]
        inputdim = dLout.shape[1]
        nchannels = dLout.shape[3]
        
        for s in range(nsamples):
            for c in range(nchannels):
                for i in range(inputdim):
                    for j in range(inputdim):
                        maxindex = np.unravel_index(np.argmax(self.input[s, i*self.stride : i*self.stride + self.filterdim, j*self.stride : j*self.stride + self.filterdim, c]), (self.filterdim, self.filterdim))
                        maxrow, maxcol = maxindex[0] + i*self.stride, maxindex[1] + j*self.stride
                        dLin[s, maxrow, maxcol, c] = dLout[s, i, j, c]
        return dLin

    def getWeights(self):
        return None

    def setWeights(self, weights, biases):
        pass

    def weightExist(self):
        return False



'''Flattening layer'''
class Flatten:
    def __init__(self):
        pass
    
    def forward(self, input):
        self.inshape = input.shape
        input = np.copy(input)
        input = np.reshape(input, (input.shape[0], np.prod(input.shape[1:])))
        return input

    def backward(self, dz, learningrate):
        # print(dz.reshape(self.inshape).shape)
        return dz.reshape(self.inshape)

    def getWeights(self):
        return None

    def setWeights(self, weights, biases):
        pass

    def weightExist(self):
        return False



'''Fully connected layer'''
class Fullyconnected:
    def __init__(self, outputdim):
        self.outputdim = outputdim
        self.isfirst = True
    
    def forward(self, input):
        inputdim = input.shape[1]
        self.input = input
        if self.isfirst == True:
            self.isfirst = False
            self.weights = np.random.randn(inputdim, self.outputdim) * np.sqrt(2/(inputdim))
            self.biases = np.zeros(self.outputdim)
        output = np.dot(input, self.weights) + self.biases
        # print(self.weights)
        # print('\n-----------------\n')
        return output
    
    def backward(self, dz, learningrate):
        dW = np.dot(self.input.T, dz)
        dB = np.sum(dz, axis = 0)
        # clip the gradient
        dW = np.clip(dW, -1, 1)
        dB = np.clip(dB, -1, 1)
        self.weights -= learningrate * dW 
        self.biases -= learningrate * dB
        return np.dot(self.weights, dz.T).T

    def getWeights(self):
        return self.weights, self.biases
    
    def setWeights(self, weights, biases):
        self.weights = weights
        self.biases = biases
    
    def weightExist(self):
        return True



'''Softmax'''
class Softmax:
    def __init__(self):
        pass
    
    def forward(self, input):
        exps = np.exp(input-np.max(input))
        return exps / np.sum(exps, axis=1).reshape(-1,1)

    def backward(self, dL, learningrate):
        return dL

    def getWeights(self):
        return None
    
    def setWeights(self, weights, biases):
        pass
    
    def weightExist(self):
        return False



'''Convolutional Neural Network'''
class CNN:
    def __init__(self):
        self.components = []
        self.info = []
        self.weights = []
        self.biases = []
    
    def buildnetwork(self):
        # first block
        self.components.append(Convolution(6, 5, 1, 1))
        self.info.append("conv1")
        self.components.append(Relu())
        self.info.append("relu1")
        self.components.append(Maxpooling(2, 2))
        self.info.append("maxpool1")
        self.components.append(Convolution(16, 5, 1, 1))
        self.info.append("conv2")
        self.components.append(Relu())
        self.info.append("relu2")
        self.components.append(Maxpooling(2, 2))
        self.info.append("maxpool2")
        self.components.append(Flatten())
        self.info.append("flatten")
        self.components.append(Fullyconnected(120))
        self.info.append("fc1")
        self.components.append(Relu())
        self.info.append("relu3")
        self.components.append(Fullyconnected(84))
        self.info.append("fc2")
        self.components.append(Relu())
        self.info.append("relu4")
        self.components.append(Fullyconnected(10))
        self.info.append("fc3")
        self.components.append(Softmax())
        self.info.append("softmax")
    
    def forward(self, input):
        # print(input.shape)
        for i in range(len(self.components)):
            input = self.components[i].forward(input)
            # print(input.shape)
        return input
    
    def backward(self, dL, learningrate):
        # print(dL.shape)
        for i in range(len(self.components)-1, -1, -1):
            dL = self.components[i].backward(dL, learningrate)
            # print(self.info[i])
            # print(dL.shape)
        return dL
    
    def predict(self, xtest):
        ypred = self.forward(xtest)
        # print(ypred.shape)
        # print(ypred)
        # print("\n\nbreak\nbreak\nbreak\n")
        # ypred = np.argmax(ypred, axis = 1)
        return ypred
    
    def getWeights(self):
        for i in range(len(self.components)):
            if self.components[i].getWeights() != None:
                weight, bias = self.components[i].getWeights()
                self.weights.append(weight)
                self.biases.append(bias)
        return self.weights, self.biases
    
    def setWeights(self, pklfilename):
        with open(pklfilename, 'rb') as file:
            allweights = pickle.load(file)
        
        weights = []
        biases = []
        for i in range(int(len(allweights)/2)):
            weights.append(allweights[i])
        
        for i in range(int(len(allweights)/2), len(allweights)):
            biases.append(allweights[i])

        layers = []
        for i in range(len(self.components)):
            if self.components[i].weightExist() == True:
                layers.append(self.components[i])

        for i in range(len(layers)):
            self.components[i].setWeights(weights[i], biases[i])

cnn = CNN()
cnn.buildnetwork()
pklfile = '1705120_model.pickle'
cnn.setWeights(pklfile)

path = 'NumtaDB_with_aug/training-d'
ypath = 'NumtaDB_with_aug/training-d.csv'
images = []

for file in os.listdir(path):
    if file.endswith('.png'):
        image = cv2.imread(os.path.join(path, file))
        # print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (28, 28))
        if len(image.shape) == 2:
            image = cv2.bitwise_not(image)
            image = np.expand_dims(image, axis = 2)
        image = (image - np.mean(image)) / np.std(image)
        images.append(image)

images = np.array(images)
ylabel = pd.read_csv(ypath)
ylabel = ylabel['digit'].values

ypred = cnn.predict(images)

log_prob_val = np.log(ypred)
# validation loss
ytrue = np.reshape(ylabel, (len(ylabel), 1))
loss = -np.sum(ytrue * log_prob_val) / len(ytrue)
# validation accuracy
ypred = np.argmax(ypred, axis=1)
acc = skm.accuracy_score(ytrue, ypred)
# f1 score of validation
f1 = skm.f1_score(ytrue, ypred, average='macro')

print('Validation loss: ', loss)
print('Validation accuracy: ', acc)
print('F1 score of validation: ', f1)

outpath = sys.argv[1]
fw = open(outpath + "/1705120_prediction.csv", 'w')

i=0
for file in os.listdir(path):
    if file.endswith('.png'):
        fw.write(file + ',' + str(np.argmax(ypred[i])) + '\n')
        i+=1
