import numpy as np
import matplotlib.pyplot as plt

#Funções de ajuda
def sigmoid(z):
    return 1 / (1 + np.e ** (-z))

def sigmoid_prime(a):
    #Estamos considerando que "a" já passou pela sigmoide, então a derivada se dá por:
    return a * (1 - a)

#Hiperparâmetros
lr = .05
momentum = .9

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
    global update_vector_1
    global update_vector_2
    '''
    Estamos calculando com momentum, que considera uma parcela do update_vector anterior para calcular o atual. Com um momentum de 0.9, adicionamos 90%
    do update_vector anterior ao update_vector atual. Assim, estaremos ganhando velocidade na direção anterior.
    '''
    output_error = a2 - y
    gradient2 = output_error * sigmoid_prime(a2) 
    update_vector_2 = np.matmul(gradient2, a1.T)
 
    hidden_error = np.matmul(w2.T, output_error) * sigmoid_prime(a2)
    gradient1 = hidden_error * sigmoid_prime(a1)
    update_vector_1 = np.matmul(gradient1, X.T)
    
        #AJUSTAR PARÂMETROS
    w1 -= (momentum * update_vector_1) + lr * update_vector_1
    w2 -= (momentum * update_vector_2) + lr * update_vector_2
    b1 -= lr * gradient1
    b2 -= lr * gradient2
    
    return output_error

#Execução do treinamento
errors = []
cost = []
step = []
update_vector_1 = 0
update_vector_2 = 0
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
