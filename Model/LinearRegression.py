import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self) -> None:
        self.learning_rate = 0.01
        self.bias = 0.0
        self.weight = 0.0
        self.iter = 500
        self.lossDict = {}

    def __call__(self, path) -> np.ndarray:
        x_train, x_test, y_train, y_test = np.load(path, allow_pickle=True)
        x_train = np.append([], x_train)
        x_test = np.append([], x_test)
        self.GradientDescent(x_train, y_train)
        self.PlotLossCurve()
        self.PlotTrainingData_and_Prediction(x_train, y_train)
        return x_train, x_test, y_train, y_test

    def computeLoss(self, y_pred, y_hat) -> float:
        Sub = y_pred - y_hat
        Sqr = Sub**2
        MSE = np.sum(Sqr) / len(y_pred)
        return MSE

    def computeGradient_b0(self, y_train, y_pred) -> float:
        Gradient = np.sum(y_pred - y_train) * 2 / len(y_train)
        return Gradient

    def computeGradient_b1(self, x_train, y_train, y_pred) -> float:
        Gradient = np.sum((y_pred - y_train) * x_train) * 2 / len(y_train)
        return Gradient

    def GradientDescent(self, x_train, y_train) -> None:
        for i in range(self.iter):
            y_pred = self.weight * x_train + self.bias
            Loss = self.computeLoss(y_pred, y_train)
            self.lossDict[i] = Loss
            Gradient_b0 = self.computeGradient_b0(y_train, y_pred)
            Gradient_b1 = self.computeGradient_b1(x_train, y_train, y_pred)
            self.weight -= self.learning_rate * Gradient_b1
            self.bias -= self.learning_rate * Gradient_b0

    def PlotLossCurve(self) -> None:
        lists = sorted(self.lossDict.items())
        x, y = zip(*lists)
        plt.plot(x, y)
        plt.show()
        plt.cla()

    def PlotTrainingData_and_Prediction(self, x_train, y_train) -> None:
        y_pred = self.weight * x_train + self.bias
        plt.plot(x_train, y_train, ".", color="blue")
        plt.plot(x_train, y_pred, ".", color="orange")
        plt.show()
        plt.cla()


if __name__ == "__main__":
    Model = LinearRegression()
    x_train, x_test, y_train, y_test = Model("./Dataset/regression_data.npy")
    y_pred = Model.weight * x_test + Model.bias
    Loss = Model.computeLoss(y_pred, y_test)
    print(Loss)
