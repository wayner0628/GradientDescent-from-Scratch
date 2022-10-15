import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self) -> None:
        self.weight = 0
        self.bias = 0
        self.learning_rate = 0.1
        self.iter = 1000
        self.lossDict = {}

    def __call__(self, path) -> np.ndarray:
        x_train, x_test, y_train, y_test = np.load(path, allow_pickle=True)
        x_train = np.append([], x_train)
        x_test = np.append([], x_test)
        self.GradientDescent(x_train, y_train)
        self.PlotLossCurve()
        self.PlotTrainingData_and_Prediction(x_train, y_train)
        return x_train, x_test, y_train, y_test

    def sigmoid(self, x) -> np.ndarray:
        sig = 1 / (1 + np.e ** (-x))
        return sig

    def computeGradient_b0(self, y_pred, y_train) -> float:
        Gradient = np.sum(self.sigmoid(y_pred) - y_train) * 2 / len(y_train)
        return Gradient

    def computeGradient_b1(self, y_pred, y_train, x_train) -> float:
        Gradient = np.sum(np.dot((self.sigmoid(y_pred) - y_train), x_train)) / len(y_train)
        return Gradient

    def computeLoss(self, y_pred, y_hat) -> float:
        pos_cost = y_hat.dot(np.log(self.sigmoid(y_pred)))
        neg_cost = (1 - y_hat).dot(np.log(1 - self.sigmoid(y_pred)))
        Loss = -((pos_cost + neg_cost)) / len(y_pred)
        return Loss

    def GradientDescent(self, x_train, y_train) -> None:
        for i in range(self.iter):
            y_pred = self.weight * x_train + self.bias
            Loss = self.computeLoss(y_pred, y_train)
            self.lossDict[i] = Loss
            Gradient_b0 = self.computeGradient_b0(y_pred, y_train)
            Gradient_b1 = self.computeGradient_b1(y_pred, y_train, x_train)
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
        plt.plot(x_train, y_train, ".", c="blue")
        plt.plot(x_train, y_pred, ".", c="orange")
        plt.show()
        plt.cla()

        plt.scatter(x_train, np.ones_like(x_train), c=y_train)
        plt.scatter(x_train, np.zeros_like(x_train), c=y_pred)
        plt.show()
        plt.cla()


if __name__ == "__main__":
    Model = LogisticRegression()
    x_train, x_test, y_train, y_test = Model("../Dataset/classification_data.npy")
    y_pred = Model.weight * x_test + Model.bias
    Loss = Model.computeLoss(y_pred, y_test)
    print(Loss)
