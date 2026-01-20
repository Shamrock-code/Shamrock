import matplotlib.pyplot as plt


def plot_fcc(nx, ny, nz):
    X = []
    Y = []
    Z = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = 2 * i + ((j + k) % 2)
                y = (3.0**0.5) * (j + (1) * (k % 2))
                z = 2 * (6.0**0.5) * k / 3.0

                X.append(x)
                Y.append(y)
                Z.append(z)
    return X, Y, Z


def plot_fcc_new(nx, ny, nz):
    X = []
    Y = []
    Z = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = 2 * i + (j % 2)
                y = (3**0.5) * j
                z = 2 * (6.0**0.5) * k / 3.0

                X.append(x)
                Y.append(y)
                Z.append(z)

    return X, Y, Z


X1, Y1, Z1 = plot_fcc(10, 10, 10)
X2, Y2, Z2 = plot_fcc_new(0, 0, 0)

plt.figure()
plt.scatter(X1, Y1)
plt.scatter(X2, Y2)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.axis("equal")

plt.figure()
plt.scatter(X1, Z1)
plt.scatter(X2, Z2)
plt.xlabel("$x$")
plt.ylabel("$z$")
plt.axis("equal")

plt.figure()
plt.scatter(Y1, Z1)
plt.scatter(Y2, Z2)
plt.xlabel("$y$")
plt.ylabel("$z$")
plt.axis("equal")

plt.show()
