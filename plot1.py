import pandas as pd
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plot

target_url = './data/glass.csv'

## 读取数据集
glass = pd.read_csv(target_url,header=None,prefix="V")
glass.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']

## 计算所有实值列（包括目标）的相关矩阵
corMat = DataFrame(glass.iloc[:,1:-1].corr())
print(corMat)

## 使用热图可视化相关矩阵
plot.pcolor(corMat)
plt.title('Correlation Matrix')
plot.savefig('./plots/correlationMatrix.pdf')
plot.show()
