import random
import numpy
import copy as cp
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import csv

class Neural_network():
  def __init__(self,layers): #layers must be passed as a vector of sizes
    self.layers = layers
    self.weights = [] #Synapses are stored as a matrix of indexes (target neuron index, source neuron index)
    self.bias = []
    self.training_step = -1

    for i in range(1,len(layers)):
      weight = []
      for j in range(layers[i]):
        weight.append(random.random())
      self.bias.append(weight)

    for i in range(len(layers)-1):
      weight = []
      for j in range(layers[i+1]):
        target_neuron = []
        for k in range(layers[i]):
          target_neuron.append(random.random())
        weight.append(target_neuron)
      self.weights.append(weight)
    return
  

  def activation_function(self, x):
    return 1/(1+numpy.exp(-0.01*x)) #sigmoid function
  
  def error(self,a,b):  #Function used to calculate error between two results (statistical purposes)
    error = 0
    for i in range(len(a)):
      error+=(a[i]-b[i])**2
    return error/len(a)

  def forward_propagation(self,inp):
    neurons = cp.deepcopy(self.bias) #first layer is not registered in bias (not needed)
    for i in range(len(self.layers)-1): #for each non-input layer
      for j in range(len(neurons[i])): #for each neuron in that layer
        x = self.bias[i][j]
        if(i != 0):
          for k in range(len(neurons[i-1])): #for each neuron immediately before it
            x = x+self.weights[i][j][k]*neurons[i-1][k]
        else: #first layer, take from input
          for k in range(len(inp)):
            x = x+self.weights[i][j][k]*inp[k]
        neurons[i][j] = self.activation_function(x)
    return neurons #Return the whole neuron activation network

  def output(self,inp):
    return self.forward_propagation(inp)[-1] #Return only the last neuron layer

  def gradient_descent(self,inp,output):
    neurons = self.forward_propagation(inp)
    sigmas = cp.deepcopy(self.bias)
    weight_cost = cp.deepcopy(self.weights)

    for i in range(len(sigmas[-1])): #Compute sigmas on last layer
      sigmas[-1][i] = (output[i]-neurons[-1][i])*neurons[-1][i]*(1-neurons[-1][i])
    for i_ in range(len(sigmas)-1): #For each layer before it (excluding the input layer) compute sigmas
      i = len(sigmas)-2-i_
      for j in range(len(sigmas[i])): #For each neuron in the layer
        sigma = 0
        for k in range(len(sigmas[i+1])): #For each neuron in the next layer and weight connecting them
          sigma += sigmas[i+1][k]*self.weights[i+1][k][j]
        sigmas[i][j] = sigma*neurons[i][j]*(1-neurons[i][j])

    for i in range(len(weight_cost)): #Compute gradient weight cost
      for j in range(self.layers[i+1]):
        for k in range(self.layers[i]):
          if i == 0:
            source_neuron = inp[k]
          else:
            source_neuron = neurons[i-1][k]
          weight_cost[i][j][k] = sigmas[i][j]*source_neuron

    error = self.error(output,neurons[-1])
    return weight_cost,sigmas,error #sigmas = bias_cost

  def train(self,inputs,outputs,batch_size=0):
    total_error = 0
    total_weight_error = []
    total_bias_error = []

    if(batch_size == 0):
      inputs = [inputs]
      outputs = [outputs]
      batch_size = 1
    #Initialize errors to 0
    for i in range(1,len(self.layers)):
      weight = []
      for j in range(self.layers[i]):
        weight.append(0)
      total_bias_error.append(weight)

    for i in range(len(self.layers)-1):
      weight = []
      for j in range(self.layers[i+1]):
        target_neuron = []
        for k in range(self.layers[i]):
          target_neuron.append(0)
        weight.append(target_neuron)
      total_weight_error.append(weight)

    for batch in range(batch_size):
      w,s,e = self.gradient_descent(inputs[batch],outputs[batch])
      total_error += e
      for i in range(len(total_bias_error)):
        for j in range(len(total_bias_error[i])):
          total_bias_error[i][j]+=s[i][j]
      for i in range(len(total_weight_error)):
        for j in range(len(total_weight_error[i])):
          for k in range(len(total_weight_error[i][j])):
            total_weight_error[i][j][k] += w[i][j][k]

    for i in range(len(self.layers)-1):
      for j in range(self.layers[i+1]):
        self.bias[i][j]-=self.training_step*total_bias_error[i][j]/batch_size
    for i in range(len(self.layers)-1):
      for j in range(self.layers[i+1]):
        for k in range(self.layers[i]):
          self.weights[i][j][k]-=self.training_step*total_weight_error[i][j][k]/batch_size
    return total_error


  def __repr__(self):
    return(f"layers\n{self.layers}\n\nbiases\n{self.bias}\n\nweights\n{self.weights}")


def load_mnist(name):
  #Process mnist dataset
  file = open(name, 'r')
  data_csv = csv.reader(file, delimiter=';')
  mnist = []
  results = []
  for row in data_csv:
      mnist.append(row)
  #Convert strings to numbers
  mnist.pop(0)
  for row in range(len(mnist)):
    mnist[row] = list(map(int, mnist[row][0].split(',')))
    results.append(mnist[row].pop(0))
    for i in range(len(mnist[row])):
      mnist[row][i] = (mnist[row][i])/255
  print("Mnist loaded")
  return mnist, results

def save_net(model,name):
  with open(f"{name}.pickle", 'wb') as p:
    pickle.dump(model, p)
  print(f"Network saved as \"{name}.pickle\"")


def load_net(name):
  with open(f"{name}.pickle", 'rb') as p:
    model = pickle.load(p)
  print(f"\"{name}\" loaded")
  return model

def open_image(name):
  final_image = []
  im = Image.open(name, 'r')
  preprocesed_im = list(im.getdata())
  for pixel in preprocesed_im:
    final_image.append(max(pixel))
  normalizer = min(final_image)
  for i in range(len(final_image)):
    final_image[i] = final_image[i]-normalizer
  normalizer = max(final_image)
  bright_ratio = 0
  for i in range(len(final_image)):
    final_image[i] = final_image[i]/normalizer
    if final_image[i] >= 0.5:
      bright_ratio+=1
  if bright_ratio/784 > 0.5:
    for i in range(len(final_image)):
      final_image[i] = 1-final_image[i]
  return final_image

def sumup_output(output):
  x = cp.deepcopy(output)
  sol1= numpy.argmax(x)
  ratio1 = x[sol1]
  x[sol1] = 0
  sol2 = numpy.argmax(x)
  ratio2 = x[sol2]
  x[sol2] = 0
  sol3 = numpy.argmax(x)
  ratio3 = x[sol3]
  if(ratio1/ratio2 < 3):
    if(ratio1/ratio3 < 4):
      print(f"Model guessed {sol1}, but it could also be {sol2} or {sol3}")
    else:
      print(f"Model guessed {sol1}, but it could also be {sol2}")
  else:
    print(f"Model guessed {sol1}")


def test_accuracy(tests, model, report_interval=0
):
  test,results = load_mnist("mnist_test.csv")
  errors = 0
  for i in range(tests):
    res = results[i]
    out = model.output(test[i])
    if(i%report_interval == 0 and report_interval != 0):
      print(f"Iteration {i}")
      im = numpy.array(res)
      img = im.reshape(28,28)
      plt.imshow(img) #cmap = "gray"
      plt.show()
      print(model.output(im))
      sumup_output(model.output(im))
    if numpy.argmax(out) != res:
      errors+=1
  print(f"{errors} errors out of {tests} tests. Error rate {errors/tests*100}%")


# To test a number: 

# im = numpy.array(open_image("test_image.png"))
# img = im.reshape(28,28)
# plt.imshow(img) #cmap = "gray"
# plt.show()
# model = load_net(f"Model [784, 512, 10], 176000 steps")
# print(model.output(im))
# sumup_output(model.output(im))






layer_architecture = [784,512,10]
saved_steps = 176000
save_interval = 1000
report_interval = 100
accuracy_report_interval = 10000
batch_size = 10 #Must divide save_interval


if(saved_steps == 0):
  model = Neural_network(layer_architecture)
else:
  model = load_net(f"Model {layer_architecture}, {saved_steps} steps")
mnist,results = load_mnist('mnist_train.csv')

model.training_step = -1

error = 0
outputs = []
inputs = []
for i in range(saved_steps+1, 240001): #4 epochs
  index = random.randint(0,60000-1)
  out = [0,0,0,0,0,0,0,0,0,0]
  out[results[index]] = 1
  inputs.append(mnist[index])
  outputs.append(out)
  if((i-1)%batch_size==0 and i!=saved_steps+1):
    error+=model.train(inputs,outputs,batch_size)
    outputs = []
    inputs = []
  if(i%report_interval == 0 and i%save_interval):
    print(f"Iteration {i}")
  if(i%save_interval == 0 and i!= saved_steps+1):
    print(f"--Iteration {i}. Current error: {error/save_interval}")
    save_net(model,f"Model {model.layers}, {i} steps")
    error = 0
  if(i%accuracy_report_interval == 0):
    print("Testing accuracy")
    test_accuracy(10000,model)