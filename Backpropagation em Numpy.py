import numpy as np
import matplotlib.pyplot as plt

#Funções de ajuda
def sigmoid(z):
    return 1 / (1 + np.e ** (-z))

def sigmoid_prime(a):
    return a * (1 - a)

def cost_prime(o, y):
    return o - y

#Hiperparâmetros
lr = .01
momentum = .8

#Dados de entrada
X = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]])
y = np.asarray([[1], [1], [0], [0]])

#Inicialização aleatória dos pesos e bias
w1 = np.random.randn(2, 2)
w2 = np.random.randn(1, 2)
b1 = np.random.randn(2, 1)
b2 = np.random.randn(1, 1)

def forward(X, w1, w2, b1, b2):
    for index in range(len(X)):
        X_i = X[index].reshape((2, 1))
        y_i = y[index].reshape((1, 1))
        z1 = np.dot(w1, X_i) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)
        print("\tPrevisão: ", a2[0][0])

def train(X, y, w1, w2, b1, b2):
    #FORWARD
    X = X.reshape((2, 1))
    y = y.reshape((1, 1))
    z1 = np.dot(w1, X) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    
    #BACK
        #ERROS NAS CAMADAS
    output_error = a2 - y
    hidden_error = np.dot(w2.T, output_error)
    
        #VETORES UPDATES
    gradient2 = output_error * sigmoid_prime(a2) 
    update_vector_2 = np.dot(gradient2, a1.T)
    gradient1 = hidden_error * sigmoid_prime(a1)
    update_vector_1 = np.dot(gradient1, X.T)
    
        #AJUSTAR PARÂMETROS
    w1 -= lr * update_vector_1
    w2 -= lr * update_vector_2
    b1 -= lr * gradient1
    b2 -= lr * gradient2
    
    return output_error

#Execução do treinamento
errors = []
cost = []
step = []
for i in range(200000):
    index = np.random.randint(len(X))
    train(X[index], y[index], w1, w2, b1, b2)
    
    if i % 1000 == 0:
        error_this_example = train(X[index], y[index], w1, w2, b1, b2)
        errors.append(error_this_example[0][0])
        cost.append(sum(np.square(errors)) / len(errors))
        step.append(i)

#Plot do custo total
plt.style.use("ggplot")
plt.figure(figsize = (14, 8))
plt.plot(step, cost)
plt.ylim(0, .4)
plt.title("Custo ao longo da sessão de treino")
plt.ylabel("Custo")
plt.xlabel("Passo de treinamento")
plt.text(0, .35, "Custo no fim do treino: {:f}".format(cost[-1]))
plt.show()

#Teste
forward(np.array([[1, 0], [0, 1], [1, 1], [0, 0]]), w1, w2, b1, b2)
