import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn2pmml import sklearn2pmml,PMMLPipeline

# Load data
X = []
with open("data_multivar.txt", 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data)
        data = np.array(X)
# print(data)
num_clusters = 4

# Plot data
plt.figure()
plt.scatter(data[:,0], data[:,1], marker='o', 
        facecolors='none', edgecolors='k', s=30)#facecolors是背景颜色
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# Train the model
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
kmeans.fit(data)

# Step size of the mesh
step_size = 0.01

# Plot the boundaries
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

# 先把类似[[1,2,3],[1,2,3]]和[[4,4,4],[5,5,5]]这样的数组展平，然后再配对
predicted_labels = kmeans.predict(np.c_[x_values.ravel(), y_values.ravel()])
# print("预测结果的前200个元素是:",predicted_labels[:300])
# Plot the results
predicted_labels = predicted_labels.reshape(x_values.shape)#这个shape相当于下面要绘制图片的高和宽
# print("预测结果reshape后的前三个元素是:",predicted_labels[:3])
# print("预测结果的形状是:",predicted_labels.shape)
plt.figure()
plt.clf()
"""
官方对imshow的第一个参数的解释：
X : array-like or PIL image_for_predict
The image_for_predict data. Supported array shapes are:

(M, N): an image_for_predict with scalar data. The data is visualized using a colormap.
(M, N, 3): an image_for_predict with RGB values (0-1 float or 0-255 int).
(M, N, 4): an image_for_predict with RGBA values (0-1 float or 0-255 int), i.e. including transparency.
The first two dimensions (M, N) define the rows and columns of the image_for_predict.

Out-of-range RGB(A) values are clipped.
"""
plt.imshow(predicted_labels, interpolation='nearest',
           extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.scatter(data[:,0], data[:,1], marker='o',
        facecolors='none', edgecolors='k', s=30)

centroids = kmeans.cluster_centers_#获取聚类中心
plt.scatter(centroids[:,0], centroids[:,1], marker='o', s=200, linewidths=3,
        color='k', zorder=10, facecolors='black')#将聚类中心画得突出一点
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
plt.title('Centoids and boundaries obtained using KMeans')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

print("模型预测结果:",kmeans.predict([(1.96,-0.09),(2.84,3.16),(2.6,0.77)]))