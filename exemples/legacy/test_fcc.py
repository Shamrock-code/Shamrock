from math import fmod
import random
import matplotlib.pyplot as plt

def get_pos(i: int,j: int,k: int):

    x = float(2*i + ((j+k) % 2))
    y = float((3.**0.5)*(j + (1./3.)*(k % 2)))
    z = float(2*(6.**0.5)*k/3)

    return (x,y,z)


def get_dim(i: int,j: int,k: int):

    

    x = 2*i + ((j+k) % 2)
    y = (3.**0.5)*(j + (1./3.)*(k % 2))
    z = 2*(6.**0.5)*k/3

    return (x,y,z)


def get_posmod(i:int, j:int, k:int , I:int, J:int, K:int):

    X,Y,Z = get_dim(I,J,K)
    x,y,z = get_pos(i,j,k)

    cx,cy,cz = get_pos(i % I,j % J,k % K)

    return (cx - fmod(x, X), cy - fmod(y, Y),cz - fmod(z, Z))


def random_test(N : int, I:int, J:int, K:int):

    S = []

    for a in range(N):
        i,j,k = random.randint(0, I *2),random.randint(0, J *2),random.randint(0, K *2)

        ex,ey,ez = get_posmod(i, j, k, I, J, K)

        S.append((ex*ex + ey*ey + ez*ez)**0.5)

    plt.hist(S)
    plt.show()