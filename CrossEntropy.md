```python
import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()
```
分类标签用one hot向量表示，1的下标就是类别号。是一个shape=(None, num_classes)向量。这种类别表示法用于函数tf.nn.softmax_cross_entropy_with_logits。
```python
labels = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [1, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0]], dtype=np.float32)
```
logits数据,就是神经网络中的 X*W 的结果。是网络的一次前向传播输出不带非线性函数的结果。
```python
logits = np.array([[1, 2, 7],
                   [3, 5, 2],
                   [6, 1, 3],
                   [8, 2, 0],
                   [3, 6, 1]], dtype=np.float32)
```
类别数：
```python
num_classes = labels.shape[1]
```
tf.nn.softmax中dim默认为-1,即,tf.nn.softmax会以最后一个维度作为一维向量计算softmax
```python
predicts = tf.nn.softmax(logits=logits, dim=-1)
```
观察softmax输出的predicts可知,softmax能够放大占比重较大的项。

注意：tf.nn.softmax函数默认（dim=-1）是对张量最后一维的shape=(p,)向量进行softmax计算，得到一个概率向量。
不同的是,tf.nn.sigmoid函数对一个张量的每一个标量元素求得一个概率。也就是说tf.nn.softmax默认针对1阶张量进行运算,
可以通过指定dim来针对1阶以上的张量进行运算,但不能对0阶张量进行运算。而tf.nn.sigmoid是针对0阶张量。


将one hot编码的标签转换为类别数字,得到一个shape=(None,)的张量。

这种类别表示法用于函数tf.nn.sparse_softmax_cross_entropy_with_logits。
```python
# 顺着列(axis=1)的方向取得所有列相应的分量的最大值的下标
# 返回的是一个shape=(len(labels),)的向量
classes = tf.argmax(labels, axis=1)
```
手动求交叉熵：
```python
labels = tf.clip_by_value(labels, 1e-10, 1.0)
predicts = tf.clip_by_value(predicts, 1e-10, 1.0)
cross_entropy = tf.reduce_sum(labels * tf.log(labels/predicts), axis=1)
```
用tf.nn.softmax_cross_entropy_with_logits求交叉熵：

labels参数的shape=(None,num_classes)，在本例中即labels。
```python
cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
```
用tf.nn.sparse_softmax_cross_entropy_with_logits求交叉熵：

其labels参数的shape=(None,)的向量，在本例中即classes。
```python
cross_entropy3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, \
                                                                labels=classes)
```
以上三个交叉熵的输出结果为shape=(None,)的相同的交叉熵向量。即,None个概率向量对的交叉熵构成的向量。

下面说说tf.nn.sigmoid_cross_entropy_with_logits：

不同于softmax系列函数是张量中向量与向量间的运算。sigmoid_cross_entropy_with_logits函数则是张量中标量与标量间的运算。
```python
z = 0.8
x = 1.3
cross_entropy4 = tf.nn.sigmoid_cross_entropy_with_logits(labels=z, logits=x)
# tf.nn.sigmoid_cross_entropy_with_logits的具体实现:
cross_entropy5 = - z * tf.log(tf.nn.sigmoid(x)) \
                 - (1-z) * tf.log(1-tf.nn.sigmoid(x))
```
cross_entropy4 与cross_entropy5 的值是相等的.
sigmoid_cross_entropy_with_logits的具体实现就是 entropy = -z\cdot log(\sigma(x))-(1-z)\cdot log(1-\sigma(x)) 
