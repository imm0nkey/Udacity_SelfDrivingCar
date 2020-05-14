
  #当scatter后面参数中数组的使用方法，如s，当s是同x大小的数组，表示x中的每个点对应s中一个大小，其他如c，等用法一样，如：
  #当scatter，如s，属性s是设置散点大小
#(1)不同大小
#导入必要的模块 
import numpy as np 
import matplotlib.pyplot as plt 
#产生测试数据 
x = np.arange(1,10) 
y = x 
fig = plt.figure() 
ax1 = fig.add_subplot(111) 
#设置标题 
ax1.set_title('Scatter Plot') 
#设置X轴标签 
plt.xlabel('X') 
#设置Y轴标签 
plt.ylabel('Y') 
#画散点图 
sValue = x*10 
ax1.scatter(x,y,s=sValue,c='r',marker='x') 
#设置图标 
plt.legend('x1') 
#显示所画的图 
plt.show()
