import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("DataSets/trainingdata.txt", delimiter=",")
data = np.asmatrix(data)

data_train = np.asmatrix([[0, 0]])
i = 0
while i < data.shape[0]:
    if data[i, 0] <= 4.11:
        data_train=np.vstack((data_train, data[i,:]))
    i += 1

data_train = data_train[1:,:]
data_train = np.hstack((np.asmatrix(np.ones(data_train.shape[0])).reshape(data_train.shape[0], 1), data_train))


class LinearRegression:

    def __init__(self, features):
        self.mean_in= 0
        self.sigma_in = 0
        self.Theta = np.asmatrix(np.zeros(features+1)).reshape(1, features+1)
        self.mean_out = 0
        self.sigma_out = 0

    def init_normalize(self, feature_matrix):
        self.mean_in = feature_matrix.mean(axis=0)
        # print(mean)
        self.sigma_in = feature_matrix.std(axis=0)
        # print(sigma)
        return self.feature_normalize(feature_matrix)

    def output_normalize(self, output_vector):
        self.mean_out = output_vector.mean(axis=0)
        self.sigma_out = output_vector.std(axis=0)
        return (output_vector-self.mean_out)/self.sigma_out

    def feature_normalize(self, feature_matrix):
        m = feature_matrix.shape[0]
        normalised = np.zeros(feature_matrix.shape)
        i = 0
        while i < m:
            normalised[i, :] = np.divide(np.subtract(feature_matrix[i, :], self.mean_in), self.sigma_in)
            i += 1
        normalised = np.hstack((np.asmatrix(np.ones(m)).reshape(m, 1), normalised))
        return normalised

    def predict(self, feature_matrix):
        return np.round(np.dot(feature_matrix, self.Theta.transpose()),decimals=2)

    def cost_function(self, feature_matrix, y, lambda1):
        m = feature_matrix.shape[0]
        forwarded = self.difference(feature_matrix, y)
        theta=self.Theta
        theta[0,0] = 0
        return np.add(np.dot(forwarded.transpose(), forwarded)/(2*m), lambda1/m*np.sum(np.multiply(theta,theta)))

    def difference(self, feature_matrix, y):
        return np.subtract(self.predict(feature_matrix), y)

    def gradient_descent(self, feature_matrix, y, iterations, alpha, lambda1):
        iterator = 0
        J = np.matrix(np.zeros(iterations).reshape(iterations,1))

        while iterator < iterations:
            m = feature_matrix.shape[0]
            forwarded = self.difference(feature_matrix, y)
            summer = np.matrix(np.zeros(self.Theta.size)).reshape(self.Theta.shape)
            # print(summer.shape)

            i = 0
            while i < m:
               # print("Forwarded"+str(feature_matrix[i,:].shape))
                summer += np.dot(forwarded[i, :], feature_matrix[i, :])
                i += 1

            theta = self.Theta
            theta[0, 0] = 0
            self.Theta -= (np.multiply(alpha/m, summer) + (lambda1/m*theta))
            iterator += 1
            #cost = self.cost_function(feature_matrix, y, lambda1)
          #  print(cost.shape)
           # J[iterator, :] = cost
         #   print(str(iterator)+" th Iteration: Cost is:"+str(J[iterator, 0]))


        #plt.plot(list(range(1, iterations+1)), J)
        #plt.show()

    def train(self, feature_matrix, y, iterations, alpha, lambda1):
        self.gradient_descent(feature_matrix, y, iterations, alpha, lambda1)
        np.save("Parameters", self.Theta)

    def test(self, feature_matrix):
        return self.predict(feature_matrix)

linear_regression = LinearRegression(1)
linear_regression.train(data_train[:,0:2], data_train[:,2], 400, 0.01, 0)
number = float(input())
if number > 4.11:
    print(8.0)
else:
    X_test = np.asmatrix([[1, number]]).reshape(1, 2)
    print(linear_regression.test(X_test))

#plt.plot(data_train[:,0:2],data_train[:,2], 'ro')




