import sys
import numpy as np

def sigmod(x):
    return 1/(1+np.exp(-x))

def forward(W1,W2,b1,b2,X,Y):
    W1=np.array(W1)
    W2=np.array(W2)
    X=np.array(X)
    Y=np.array(Y)

    Z1=sigmod(np.dot(W1,X)+b1)
    Z2=sigmod(np.dot(W2,Z1)+b2)
    h=Z2
    loss=-Y*np.log(h)-(1-Y)*np.log(1-h)
    # print(f'[debug] Z1: {Z1}, Z2: {Z2}, h: {h}, loss: {loss}')
    total_loss=np.sum(loss)
    return total_loss

if __name__=='__main__':
    parts=input().strip().split(',')
    # parts='[[0.1,0.2],[0.3,0.4]],[[0.5,0.6],[0.7,0.8]],0.1,[[0.05],[0.10]],[[1],[0]]'.split(',')
    W1=eval(','.join(parts[:4]).strip())
    W2=eval(','.join(parts[4:8]).strip())
    b1=b2=eval(parts[8].strip())
    X=eval(','.join(parts[9:11]).strip())
    Y=eval(','.join(parts[11:]).strip())
    loss=forward(W1,W2,b1,b2,X,Y)
    # print(f'[debug] W1: {W1}, W2: {W2}, b1: {b1}, b2: {b2}, X: {X}, Y: {Y}')
    print(f'{loss:.3f}')

