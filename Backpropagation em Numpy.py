import numpy as np
import matplotlib.pyplot as plt

#Funções de ajuda
def sigmoid(z):
    return 1 / (1 + np.e ** (-z))

def sigmoid_prime(a):
    #Estamos considerando que "a" já passou pela sigmoide, então a derivada se dá por:
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
    '''
    Calculamos o erro na última camada e distribuímos esse erro nos weights das camadas anteriores. 
    O erro do output será a2 - y. Lembrando, Cost = sum(error ** 2). Calculamos o gradiente do output multiplicando dC/derror * derror/da2 * da2/dz2. Esse 
    gradiente será multiplicado por dz2/dW2 transposta (já que é um passo backpropagation) para achar o update vector em seguida. 
    
    dC/derror = d(error ** 2)/derror = 2*error = error = output_error.
    derror/da2 = d(a2 - y)/da2 = 1
    da2/dz2 = sigmoid_prime(a2)
    dz2/dW2 = d(W2*a1 + b2)/dW2 = a1
    '''
    output_error = a2 - y
    gradient2 = output_error * sigmoid_prime(a2) 
    update_vector_2 = np.matmul(gradient2, a1.T)
    '''Enquanto as duas primeiras derivadas são element-wise, a última é uma matmul dos gradientes com a transposta da ativação anterior
    dCost/da2 * da2/dz2 x dz2/dw2 = output_error * sigmoid_prime(a2) x a1.T
    Assim, achamos o quanto os weights tem que ser ajustados para diminuir o cost.'''
    
    '''
    Agora precisamos calcular o hidden error. Podemos pensar no hidden error de cada node em hidden como uma parcela que os pesos de cada hidden_node contribuem
    para o output_error. Pensando assim, podemos multiplicar os pesos de cada hidden_node, ou seja W2, pelo output_error. O certo seria dividir cada W pela soma
    de todos os W em W2 para termos realmente uma proporção (ex.: w0,0 / (w0,0 + w0,1) * output_error), mas podemos ignorar esse denominador já que iremos 
    multiplicar tudo por um learning_rate de qualquer maneira.
    Outra maneira é pensar nas derivadas parciais. Agora que temos o gradient2, podemos calcular o quanto a camada hidden influencia nesse gradient2, lembrando
    que o próprio gradient2 influencia no erro final. Então precisamos calcular dC/dz1, que pela regra da cadeia se dá por:
    dC/derror * derror/da2 * da2/dz2 * dz2/da1 * da1/dz1
    Já temos quase todas as parcelas dessa conta vindas de gradient2, então temos gradient2 * dz2/da1 * da1/dz1, onde:
    dz2/da1 = d(W2*a1 + b2)/da1 = W2
    da1/dz1 = sigmoid_prime(a1)
    No fim, organizando para as dimensões fecharem, temos que hidden_error é igual a W2.T * gradient2 * sigmoid_prime(a1). Substituíndo gradient2, ficamos
    com:
    hidden_error = W2.T * output_error * sigmoid_prime(a2) * sigmoid_prime(a1)
    Aqui, consideraremos os 3 primeiros termos como hidden_error e calcularemos a multiplicação por sigmoid_prime(a1) para acharmos gradient1
    Como é um passo no backprop, transpomos a matriz dos weights. 
    Agora que temos gradient1, multiplicamos por dz1/dW1 para finalmente acharmos o update vector, sabendo que:
    dz1/dW1 = d(W1*X + b1)/dW1 = X
    Como é um passo backprop, transpomos a matriz de input X.
    '''
    hidden_error = np.matmul(w2.T, output_error) * sigmoid_prime(a2)
    gradient1 = hidden_error * sigmoid_prime(a1)
    update_vector_1 = np.matmul(gradient1, X.T)
    
    '''
    Para os biases, o delta (o quanto devem ser ajustados) é simplesmente o gradiente já calculado, pois dz2/db2 = 1.
    '''
    
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
