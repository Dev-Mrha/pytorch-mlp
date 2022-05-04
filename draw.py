import matplotlib.pyplot as plt
import ast

with open('data.txt',"r") as f:
    print("here?")
    x = f.readline()
    x_list = ast.literal_eval(x)
    y = f.readline()
    y_list = ast.literal_eval(y)
    z = f.readline()
    z_list = ast.literal_eval(z)
    le = [i for i in range(len(x_list))]
    plt.figure()
    plt.plot(le, x_list)
    plt.plot(le, y_list)
    plt.plot(le, z_list)
    plt.legend(['mlp3-SGD-0.01', 'mlp3-Adam-0.01', 'mlp3-RMSprop-0.01'], loc='best')
    plt.show()
