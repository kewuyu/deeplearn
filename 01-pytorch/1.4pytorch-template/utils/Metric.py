class Metric:
    def __init__(self, correct, total):
        self.correct = correct  # 正确的预测数量
        self.total = total  # 总样本数

    def accuracy(self):
        # 直接计算准确度
        return self.correct / self.total if self.total != 0 else 0.0
