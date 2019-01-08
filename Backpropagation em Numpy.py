import numpy as np
import matplotlib.pyplot as plt

#Funções de ajuda
def sigmoid(z):
    return 1 / (1 + np.e ** (-z))

def sigmoid_prime(a):
    return (a) * (1 - (a))

#Hiperparâmetros
lr = .1

#Dados de entrada
X = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]])
y = np.asarray([[1], [1], [0], [0]])

#Arquitetura
n_inputs = 2
n_hidden = 4
n_output = 1

#Inicialização aleatória dos pesos e bias
w1 = np.random.randn(n_hidden, n_inputs)
w2 = np.random.randn(n_output, n_hidden)
b1 = np.ones((n_hidden, 1))
b2 = np.ones((n_output, 1))

def forward(X, w1, w2, b1, b2):
    for index in range(len(X)):
        X_i = X[index].reshape((n_inputs, 1))
        y_i = y[index].reshape((n_output, 1))
        z1 = np.matmul(w1, X_i) + b1
        a1 = sigmoid(z1)
        z2 = np.matmul(w2, a1) + b2
        a2 = sigmoid(z2)
        print("\tPrevisão: ", a2[0][0])

def train(X, y, w1, w2, b1, b2):
    #FORWARD
    X = X.reshape((n_inputs, 1))
    y = y.reshape((n_output, 1))
    z1 = np.matmul(w1, X) + b1
    a1 = sigmoid(z1)
    z2 = np.matmul(w2, a1) + b2
    a2 = sigmoid(z2)
    
    #BACK
    
        #ERROS NAS CAMADAS
    '''
    Calculamos o erro na última camada e distribuímos esse erro nos weights das camadas anteriores. 
    Como é um passo no backprop, transpomos a matriz dos weights. Assim, temos o erro na camada anterior.'''
    output_error = a2 - y
    hidden_error = np.matmul(w2.T, output_error)
    
        #VETORES UPDATES
    '''
    Se cost = sum(a2 - y) **2 / n, então a dCost/da2 é 2/n*(a2 - y). Podemos ignorar o 2/n, ficando a2 - y como derivada, que é o output_error
        dCost/da2 = a2 - y. 
        da2/dz2 = sigmoid_prime(a2). 
        dz2/dw2 = a1.T.
    Enquanto as duas primeiras derivadas são element-wise, a última é uma matmul dos gradientes com a transposta da ativação anterior
        dCost/da2 * da2/dz2 x dz2/dw2 = output_error * sigmoid_prime(a2) x a1.T
    Assim, achamos o quanto os weights tem que ser ajustados para diminuir o cost.
    Para os biases, o delta (o quanto devem ser ajustados) é simplesmente o gradiente já calculado, pois dz2/db2 = 1.'''
    gradient2 = (a2 - y) * sigmoid_prime(a2) 
    update_vector_2 = np.matmul(gradient2, a1.T)
    gradient1 = hidden_error * sigmoid_prime(a1)
    update_vector_1 = np.matmul(gradient1, X.T)
    
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
for i in range(100000):
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
