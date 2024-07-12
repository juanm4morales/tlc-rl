from math import ceil, log2

class Discrete:
    def __init__(self, I, M):
        if I>M:
            I=M
            print("The I(nterval) value is greater than M(ax) value. \"I\" was setted with \"M\" value.")
        self.I=I-1
        self.M=M
        self.L=self.I/self.M
        self.F=(1-self.L)*(self.I/(log2(M/(M*self.L)+1)))
             
    def log_interval(self, x):
        interval = ceil(self.F*(log2(x/(self.M*self.L)+1))+pow(self.L,2)*x)
        return interval

