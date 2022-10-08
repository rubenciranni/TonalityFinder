import math as mt 
import random as rd

dataset = [
 [2, 0, 3, 0, 4, 4, 0, 3, 0, 3, 0, 1, 1],
 [6, 0, 2, 1, 0, 2, 0, 4, 2, 0, 0, 3, 0], 
 [3, 0, 1, 0, 3, 3, 0, 4, 0, 1, 0, 1, 1],
 [3, 0, 2, 3, 0, 2, 0, 3, 1, 1, 1, 3, 0],
 [4, 0, 6, 0, 7, 4, 0, 4, 0, 4, 0, 0, 1],
 [4, 0, 5, 4, 0, 4, 0, 1, 1, 0, 1, 1, 0],
 [4, 0, 3, 0, 2, 4, 0, 3, 0, 4, 3, 2, 1],
 [2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1],
 [5, 0, 2, 0, 3, 0, 0, 1, 0, 0, 0, 0, 1],
 [7, 0, 1, 0, 1, 2, 0, 2, 0, 2, 0, 1, 1],
 [2, 0, 2, 0, 3, 2, 0, 4, 0, 1, 0, 1, 1],
 [3, 0, 2, 0, 3, 1, 0, 3, 0, 3, 0, 2, 1],
 [9, 0, 3, 0, 2, 5, 0, 2, 0, 2, 2, 0, 1],
 [3, 0, 3, 0, 5, 2, 0, 2, 0, 1, 0, 0, 1],
 [3, 1, 3, 5, 1, 5, 2, 7, 0, 0, 2, 0, 0],
 [4, 0, 2, 2, 0, 2, 0, 8, 1, 0, 0, 4, 0],
 [4, 0, 4, 4, 0, 4, 0, 8, 0, 0, 0, 0, 0],
 [4, 0, 4, 3, 0, 5, 0, 4, 1, 0, 0, 2, 0],
 [6, 0, 2, 1, 0, 2, 0, 4, 2, 0, 0, 3, 0],
 [8, 0, 9, 5, 0, 2, 0, 5, 1, 0, 0, 2, 0],
 [4, 0, 3, 4, 0, 8, 0, 6, 2, 0, 0, 3, 0]
 ]


rd.seed(1)

def sigmoide(x):
    return 1/(1+mt.exp(-x))

def RN(m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12):
    z = (m1*w1 +
         m2*w2 +
         m3*w3 +
         m4*w4 +
         m5*w5 +
         m6*w6 +
         m7*w7 +
         m8*w8 +
         m9*w9 +
         m10*w10 +
         m11*w11 +
         m12*w12 +
         b)
    return sigmoide(z)

  

#definisco la derivata della funzione sigmoide
def dsigmoide(x):
    return sigmoide(x)*(1 - sigmoide(x)) 

def train():

    #pesi inizializzati inizialmente in modo casuale
    w1 = rd.random()
    w2 = rd.random()
    w3 = rd.random()
    w4 = rd.random()
    w5 = rd.random()
    w6 = rd.random()
    w7 = rd.random()
    w8 = rd.random()
    w9 = rd.random()
    w10 = rd.random()
    w11 = rd.random()
    w12 = rd.random()
    b = rd.random()     

    epochs = 20000 #imposto il numero di iterazioni
    learning_rate = 1/10 #imposto il learning rate
    
    for i in range(epochs):
        if i % 2000 == 0:
            print('=',end='')
        elif i == (epochs - 1):
            print('')
            print('allenamento completato\n')
       
        point = dataset[rd.randint(0,len(dataset)-1)] # prendo un elemento casuale dal dataset
        
        z = (int(point[0])*w1 +
             int(point[1])*w2 +
             int(point[2])*w3 +
             int(point[3])*w4 +
             int(point[4])*w5 +
             int(point[5])*w6 +
             int(point[6])*w7 +
             int(point[7])*w8 +
             int(point[8])*w9 +
             int(point[9])*w10 +
             int(point[10])*w11 +
             int(point[11])*w12 +
             b)
        
        pred = sigmoide(z) # previsione della rete
        
        target = int(point[12]) #il valore target
        
        # funzione di costo (errore quadratico) del punto casuale attuale
        cost = (pred - target)**2
        
        #CALCOLO DELLE DERIVATE        
        dz_dw1 = int(point[0]) #derivata parziale di z rispetto a w1
        dz_dw2 = int(point[1])#derivata parziale di z rispetto a w2
        dz_dw3 = int(point[2])
        dz_dw4 = int(point[3])
        dz_dw5 = int(point[4])
        dz_dw6 = int(point[5])
        dz_dw7 = int(point[6])
        dz_dw8 = int(point[7])
        dz_dw9 = int(point[8])
        dz_dw10 = int(point[9])
        dz_dw11 = int(point[10])
        dz_dw12 = int(point[11])
        dz_db = 1         #derivata parziale di z rispetto a b
        
        dcost_dpred = 2 * (pred - target) #derivata parziale del costo rispetto alla previsione
        dpred_dz = dsigmoide(z) #derivata parziale della previsione rispetto a z
        
        dcost_dz = dcost_dpred * dpred_dz #derivata parziale di z rispetto alla previsione (derivata di una funzione composta)
        

        dcost_dw1 = dcost_dz * dz_dw1 #derivata parziale del costo rispetto a w1
        dcost_dw2 = dcost_dz * dz_dw2 #derivata parziale del costo rispetto a w2
        dcost_dw3 = dcost_dz * dz_dw3
        dcost_dw4 = dcost_dz * dz_dw4
        dcost_dw5 = dcost_dz * dz_dw5
        dcost_dw6 = dcost_dz * dz_dw6
        dcost_dw7 = dcost_dz * dz_dw7
        dcost_dw8 = dcost_dz * dz_dw8
        dcost_dw9 = dcost_dz * dz_dw9
        dcost_dw10 = dcost_dz * dz_dw10
        dcost_dw11 = dcost_dz * dz_dw11
        dcost_dw12 = dcost_dz * dz_dw12
        dcost_db = dcost_dz * dz_db #derivata parziale del costo rispetto a b
        
        #aggiornamento dei pesi e del bias
        w1 = w1 - learning_rate * dcost_dw1
        w2 = w2 - learning_rate * dcost_dw2
        w3 = w3 - learning_rate * dcost_dw3
        w4 = w4 - learning_rate * dcost_dw4
        w5 = w5 - learning_rate * dcost_dw5
        w6 = w6 - learning_rate * dcost_dw6
        w7 = w7 - learning_rate * dcost_dw7
        w8 = w8 - learning_rate * dcost_dw8
        w9 = w9 - learning_rate * dcost_dw9
        w10 = w10 - learning_rate * dcost_dw10
        w11 = w11 - learning_rate * dcost_dw11
        w12 = w12 - learning_rate * dcost_dw12
        b = b - learning_rate * dcost_db
        
    return w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, b


#definisce i pesi e il bias aggiornati 
w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, b = train()
print('w1:',w1)
print('w2:',w2) 
print('w3:',w3) 
print('w4:',w4) 
print('w5:',w5)
print('w6:',w6)
print('w7:',w7)
print('w8:',w8) 
print('w9:',w9) 
print('w10:',w10) 
print('w11:',w11) 
print('w12:',w12) 
print('b:',b)
print('\n')


#interfaccia
M=[0,0,0,0,0,0,0,0,0,0,0,0]
t = input('inserisci la tonica (# diesis; b bemolle): ').lower().strip()
tonalità = {'do':0, 'do#':1, 'reb':1, 're':2, 're#':3, 'mib':3, 'mi':4, 'fa':5, 'fa#':6,
            'solb':6, 'sol':7, 'sol#':8, 'lab':8, 'la':9, 'la#':10, 'sib':10, 'si':11}
    
print('inserisci le note che compongono la melodia ( digita * per terminare): ')
nota=input('-> ').strip().lower() 
while nota != '*':      
    if nota == 'do' or nota == 'si#':
        M[0-tonalità[t]]+=1
    if nota == 'do#' or nota == 'reb':
        M[1-tonalità[t]]+=1
    if nota == 're':
        M[2-tonalità[t]]+=1
    if nota == 're#' or nota == 'mib':
        M[3-tonalità[t]]+=1
    if nota == 'mi':
        M[4-tonalità[t]]+=1
    if nota == 'fa' or nota == 'mi#':
        M[5-tonalità[t]]+=1
    if nota == 'fa#' or nota == 'solb':
        M[6-tonalità[t]]+=1
    if nota == 'sol':
        M[7-tonalità[t]]+=1
    if nota == 'sol#' or nota == 'lab':
        M[8-tonalità[t]]+=1
    if nota == 'la':
        M[9-tonalità[t]]+=1
    if nota == 'la#' or nota == 'sib':
        M[10-tonalità[t]]+=1
    if nota == 'si':
        M[11-tonalità[t]]+=1
    nota=input('-> ').strip().lower()

print('lista delle frequenze: ',M)
prediction = RN(M[0], M[1], M[2], M[3], M[4], M[5], M[6], M[7], M[8], M[9], M[10], M[11])
print('RN =',prediction)

if prediction <= 0.5: 
    print('Tonalità minore, la melodia ha un carattere triste')
else: 
    print('Tonalità maggiore, la melodia ha un carattere allegro') 
