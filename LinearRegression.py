import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self) -> None:
        self.learning_rate = 0.01
        self.bias = 0.0
        self.weight = 0.0
        self.iter = 500
        self.lossDict = {}

    def __call__(self) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = np.load("regression_data.npy", allow_pickle=True)
        self.x_train = np.append([], self.x_train)
        self.x_test = np.append([], self.x_test)
        self.GradientDescent()

    def computeLoss(self, y_pred, y_hat) -> float:
        Sub = y_pred - y_hat
        Sqr = Sub**2
        MSE = np.sum(Sqr) / len(y_pred)
        return MSE

    def computeGradient_b0(self) -> float:
        Gradient = np.sum(self.y_pred - self.y_train) * 2 / len(self.x_train)
        return Gradient

    def computeGradient_b1(self) -> float:
        Gradient = np.sum(np.dot((self.y_pred - self.y_train), self.x_train)) * 2 / len(self.x_train)
        return Gradient

    def GradientDescent(self):
        for i in range(self.iter):
            self.y_pred = self.weight * self.x_train + self.bias
            Loss = self.computeLoss(self.y_pred, self.y_train)
            self.lossDict[i] = Loss
            Gradient_b0 = self.computeGradient_b0()
            Gradient_b1 = self.computeGradient_b1()
            self.weight -= self.learning_rate * Gradient_b1
            self.bias -= self.learning_rate * Gradient_b0

    def PlotLossCurve(self) -> None:
        lists = sorted(self.lossDict.items())
        x, y = zip(*lists)
        plt.plot(x, y)
        plt.show()
        plt.cla()

    def PlotTrainingData_and_Prediction(self) -> None:
        plt.plot(self.x_train, self.y_train, ".", color="blue")
        plt.plot(self.x_train, self.y_pred, ".", color="orange")
        plt.show()
        plt.cla()

    def PrintTestingLoss(self) -> None:
        y_pred = self.weight * self.x_test + self.bias
        Loss = self.computeLoss(y_pred, self.y_test)
        print(Loss)


Model = LinearRegression()
Model()
Model.PlotLossCurve()
Model.PlotTrainingData_and_Prediction()
Model.PrintTestingLoss()
