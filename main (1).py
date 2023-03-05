import csv
import matplotlib.pyplot as plt
import numpy as np


# read data from csv file
def read_data():
    t=[]
    y=[]
    i=-1
    with open('GOOGL.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            i+=1
            if(i==0):
                continue

            t.append(i)
            y.append(float(row[1]))
    return t,y

# calculate and print error of last 10 data
def print_errors(y,y_pred):
    N=len(y)
    for i in range(10):
        j=N-10+i
        print('')
        print('Day #'+str(j+1))
        print('calculated value: '+'{:.2f}'.format(y_pred[j]))
        print('actual value: '+'{:.2f}'.format(y[j]))
        print('error: '+'{:.2f}'.format(y_pred[j]-y[j]))


def plot_results(t,y,y_pred):
    # Draw real data with red hollow circle
    plt.plot(t,y,'o', markerfacecolor='none', markeredgecolor='r')
    # Draw a model prediction with a blue line
    plt.plot(t,y_pred,'-', color='b')
    # horizontal axis
    plt.xlabel('Day')
    # vertical axis
    plt.ylabel('Open')
    plt.legend(['Data', 'Regression'])
    plt.show()

if __name__ == '__main__':

    # reading data and save them in t and y
    data=read_data()
    t=data[0]
    y=data[1]

    # linear regression
    N=len(t)
    A11=0; A12=0
    A21=0; A22=0
    b1=0; b2=0
    for i in range(N-10):
        A11 += 1; A12 += t[i]
        A21 += t[i]; A22 += t[i]**2
        b1 += y[i]
        b2 += t[i]*y[i]

    A = np.array([[A11, A12], [A21, A22]])
    b = np.array([b1, b2])
    x = np.linalg.solve(A, b)

    print('Linear Regression:')
    print('y = '+"{:.4f}".format(x[0])+' + '+"{:.4f}".format(x[1])+' t')

    # predict based on the linear regression for all data
    y_pred=[]
    for i in range(N):
        y_pred.append(x[0]+x[1]*t[i])

    print_errors(y,y_pred)
    plot_results(t,y,y_pred)

    # quadratic regression
    A11=0; A12=0; A13=0
    A21=0; A22=0; A23=0
    A31=0; A32=0; A33=0
    b1=0;  b2=0;  b3=0
    for i in range(N-10):
        A11+=1;       A12+=t[i];    A13+=t[i]**2
        A21+=t[i];    A22+=t[i]**2; A23+=t[i]**3
        A31+=t[i]**2; A32+=t[i]**3; A33+=t[i]**4
        b1+=y[i]
        b2+=t[i]*y[i]
        b3+=t[i]**2*y[i]

    A = np.array([[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]])
    b = np.array([b1, b2, b3])
    x = np.linalg.solve(A, b)

    print('Quadratic Regression:')
    print('y = '+"{:.4f}".format(x[0])+' + '+"{:.4f}".format(x[1])+' t + '+"{:.4f}".format(x[2])+' t^2')

    # predict based on the linear regression for all data
    y_pred = []
    for i in range(N):
        y_pred.append(x[0]+x[1]*t[i]+x[2]*t[i]**2)

    print_errors(y, y_pred)
    plot_results(t, y, y_pred)
