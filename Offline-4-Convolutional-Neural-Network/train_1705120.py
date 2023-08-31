import numpy as np
import os
import cv2
import pandas as pd
import sklearn.metrics as skm
import matplotlib.pyplot as plt
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
    
    def train(self, xtrain, ytrain, learningrate):
        yhat = self.forward(xtrain)
        # print(yhat.shape, ytrain.shape)
        dL = yhat - ytrain
        self.backward(dL, learningrate)
        # print("\nbreak\n")
        return yhat
    
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
        for i in range(len(allweights)/2):
            weights.append(allweights[i])
        
        for i in range(len(allweights)/2, len(allweights)):
            biases.append(allweights[i])

        layers = []
        for i in range(len(self.components)):
            if self.components[i].getWeights() != None:
                layers.append(self.components[i])

        for i in range(len(layers)):
            self.components[i].setWeights(weights[i], biases[i])



'''Training'''

path = 'NumtaDB_with_aug/training-a'
ypath = 'NumtaDB_with_aug/training-a.csv'
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
images = images[:5000]
ylabel = pd.read_csv(ypath)
ylabel = ylabel['digit'].values
ylabel = ylabel[:5000]

cnn = CNN()
cnn.buildnetwork()
# cnn.train(images[:nsamples], ytrain[:nsamples], 0.001, 5)

# pool = maxpooling(3, 3)
# output = pool.forward(images[0])
# print(output.shape)

# take batch of 15 samples to train each time

nepochs = 7
batch = 15
nbatches = len(images) // batch
nvalidation = int(len(images)*0.2)

epochlist = []
trainlosslist = []
trainacculist = []
f1trainlist = []

validationlosslist = []
validationacculist = []
f1validationlist = []

# open('results.txt', 'w').close()
# fw = open('results-00005.txt', 'a')
# fw.write("============================================ rate: 00005 ============================================\n")
# fw.close()
for epoch in range(nepochs):
    print("epoch: ", epoch+1)
    epochlist.append(epoch+1)
    ypredlist = []
    for i in range(0, len(images), batch):
        if (i+batch >= len(images)):
            xtrain = images[i:]
            ytrue = ylabel[i:]
        else:
            xtrain = images[i:i+batch]
            ytrue = ylabel[i:i+batch]
        ytrain = np.zeros((len(ytrue), 10))
        rows = np.arange(len(ytrain))
        ytrain[rows, ytrue] = 1
        yhat = cnn.train(xtrain, ytrain, 0.00005)
        ypredlist.extend(yhat)
    # per epoch f1-score, train accuracy, train loss, test accuracy, validation accuracy, validation loss
    
    # softmax train loss
    log_prob_train = np.log(ypredlist)
    # training loss
    ytrain = np.reshape(ylabel, (len(ylabel), 1))
    train_loss = -np.sum(ytrain * log_prob_train) / len(ylabel) 
    trainlosslist.append(train_loss)
    # train accuracy
    ypredlist = np.argmax(ypredlist, axis=1)
    train_acc = skm.accuracy_score(ylabel, ypredlist)
    trainacculist.append(train_acc)
    # f1-score of train
    f1_train = skm.f1_score(ylabel, ypredlist, average='macro')
    f1trainlist.append(f1_train)

    # validation
    validx = images[:nvalidation]
    ytrue = ylabel[:nvalidation]
    ypred = cnn.predict(validx)
    log_prob_val = np.log(ypred)
    # validation loss
    ytrue = np.reshape(ytrue, (len(ytrue), 1))
    val_loss = -np.sum(ytrue * log_prob_val) / len(ytrue)
    validationlosslist.append(val_loss)
    # validation accuracy
    ypred = np.argmax(ypred, axis=1)
    val_acc = skm.accuracy_score(ytrue, ypred)
    validationacculist.append(val_acc)
    # f1 score of validation
    f1_validation = skm.f1_score(ytrue, ypred, average='macro')
    f1validationlist.append(f1_validation)

    print(f'Epoch {epoch+1}/{nepochs}: Train Accuracy: {train_acc}')
    print(f'Epoch {epoch+1}/{nepochs}: Train Loss: {train_loss}')
    print(f'Epoch {epoch+1}/{nepochs}: Validation Loss: {val_loss}')
    print(f'Epoch {epoch+1}/{nepochs}: Validation Accuracy: {val_acc}')
    print(f'Epoch {epoch+1}/{nepochs}: F1 Score Train: {f1_train}')
    print(f'Epoch {epoch+1}/{nepochs}: F1 Score Validation: {f1_validation}')

    # with open('results-00005.txt', 'a') as f:
    #     f.write(f'Epoch {epoch+1}/{nepochs}: Train Accuracy: {train_acc}\n')
    #     f.write(f'Epoch {epoch+1}/{nepochs}: Train Loss: {train_loss}\n')
    #     f.write(f'Epoch {epoch+1}/{nepochs}: Validation Loss: {val_loss}\n')
    #     f.write(f'Epoch {epoch+1}/{nepochs}: Validation Accuracy: {val_acc}\n')
    #     f.write(f'Epoch {epoch+1}/{nepochs}: F1 Score Train: {f1_train}\n')
    #     f.write(f'Epoch {epoch+1}/{nepochs}: F1 Score Validation: {f1_validation}\n')
    #     f.write("\n")
    
    print(ypred)
    print(ytrue)


'''Plotting'''
# -------------------------------------TRAIN------------------------------------ #
# Train Loss vs Epochs
plt.plot(epochlist, trainlosslist)
plt.xlabel('epoch')
plt.ylabel('train_loss')
plt.title('Plot of epoch vs train_loss')
# plt.legend(['blue'])
plt.savefig("train_loss-00005.png")
# plt.show()

plt.cla()
plt.clf()
# Train Accuracy vs Epochs
plt.plot(epochlist, trainacculist)
plt.xlabel('epoch')
plt.ylabel('train_accuracy')
plt.title('Plot of epoch vs train_accuracy')
# plt.legend(['blue'])
plt.savefig("train_accuracy-00005.png")
# plt.show()


plt.cla()
plt.clf()
# f1-score of train vs epochs
plt.plot(epochlist, f1trainlist)
plt.xlabel('epoch')
plt.ylabel('f1-score_train')
plt.title('Plot of epoch vs f1-score_train')
plt.legend(['blue'])
plt.savefig("f1-score_train-00005.png")
# plt.show()
# ----------------------------------VALIDATION----------------------------------- #

plt.cla()
plt.clf()
# Validation Loss vs Epochs
plt.plot(epochlist, validationlosslist)
plt.xlabel('epoch')
plt.ylabel('validation_loss')
plt.title('Plot of epoch vs validation_loss')
plt.legend(['blue'])
plt.savefig("validation_loss-00005.png")
# plt.show()


plt.cla()
plt.clf()
# Validation Accuracy vs Epochs
plt.plot(epochlist, validationacculist)
plt.xlabel('epoch')
plt.ylabel('validation_accuracy')
plt.title('Plot of epoch vs validation_accuracy')
plt.legend(['blue'])
plt.savefig("validation_accuracy-00005.png")
# plt.show()


plt.cla()
plt.clf()
# f1-score of validation vs epochs
plt.plot(epochlist, f1validationlist)
plt.xlabel('epoch')
plt.ylabel('f1-score_validation')
plt.title('Plot of epoch vs f1-score_validation')
plt.legend(['blue'])
plt.savefig("f1-score_validation-00005.png")
# plt.show()

#---------------------------------Save Model-------------------------------------#
picklefile = "1705120_model.pickle"
weights, biases = cnn.getWeights()
all_weights = []
for w in weights:
    all_weights.append(w)
for b in biases:
    all_weights.append(b)    
with open(picklefile, 'wb') as f:
    pickle.dump(all_weights, f)
