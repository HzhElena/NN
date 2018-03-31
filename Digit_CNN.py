import tensorflow as tf
from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()
X_data = digits.data.astype(np.float32)
Y_data = digits.target.astype(np.float32).reshape(-1,1)
print (X_data.shape)
print (Y_data.shape)

# 对数据进行预处理
#最小-最大规范化对原始数据进行线性变换 变换到[0,1]区间
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
scaler = MinMaxScaler()
X_data = scaler.fit_transform(X_data)
Y = OneHotEncoder().fit_transform(Y_data).todense() # one-hot 编码
# 把 X 转换成 图片的格式
X = X_data.reshape(-1,8,8,1)
print (X.shape)

#生成 batch
batch_size = 8
def generate_batch(X,Y,n_examples,batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i * batch_size
        end = start + batch_size
        batch_xs = X[start:end]
        batch_ys = Y[start:end]
        yield batch_xs,batch_ys
       
tf.reset_default_graph()
# 输入层
#输入4维 [batch_size 一个batch的图片数量, in_height, in_width, in_channels图像通道数]
tf_X = tf.placeholder(tf.float32,[None,8,8,1]) 
tf_Y = tf.placeholder(tf.float32,[None,10])
# 卷积层 + 激活层
# 卷积核是一个4维格式的数据：shape表示为：[height,width,in_channels, out_channels输出 feature map的个数]
conv_filter_w1 = tf.Variable(tf.random_normal([3,3,1,10]))
conv_filter_b1 = tf.Variable(tf.random_normal([10]))
# strides：表示步长：一个长度为4的一维列表
# strides = [batch , in_height , in_width, in_channels]。其中 batch 和 in_channels 要求一定为1
#tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
relu_feature_maps1 = tf.nn.relu(tf.nn.conv2d(tf_X,conv_filter_w1,strides=[1,1,1,1],padding = 'SAME')+conv_filter_b1)
# 池化层
#value：表示池化的输入：一个4维格式的数据，数据的 shape 由 data_format 决定，默认情况下shape 为[batch, height, width, channels]
#ksize：表示池化窗口的大小：一个长度为4的一维列表，一般为[1, height, width, 1]，因不想在batch和channels上做池化，则将其值设为1。
max_pool1 = tf.nn.max_pool(relu_feature_maps1,ksize = [1,3,3,1],strides=[1,2,2,1],padding="SAME")
print(max_pool1)

#卷积层
# input 10 feature maps , output 5 feature maps
conv_filter_w2 = tf.Variable(tf.random_normal([3,3,10,5]))
conv_filter_b2 = tf.Variable(tf.random_normal([5]))
conv_out2 = tf.nn.conv2d(relu_feature_maps1, conv_filter_w2,strides=[1, 2, 2, 1], padding='SAME') + conv_filter_b2
print (conv_out2)

# batch normalization 归一化层+激活层 
# tf.nn.moments(x, axes, name=None, keep_dims=False) axes=[0]表示按列计算
batch_mean, batch_var = tf.nn.moments(conv_out2, [0, 1, 2], keep_dims=True)
shift = tf.Variable(tf.zeros([5]))
scale = tf.Variable(tf.ones([5]))
epsilon = 1e-3
BN_out = tf.nn.batch_normalization(conv_out2, batch_mean, batch_var, shift, scale, epsilon)
print (BN_out)
relu_BN_maps2 = tf.nn.relu(BN_out)

# 池化层
max_pool2 = tf.nn.max_pool(relu_BN_maps2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
print(max_pool2)

# 将特征图进行展开
max_pool2_flat = tf.reshape(max_pool2, [-1, 2*2*5])
print (max_pool2_flat)

# 全连接层
# 一层 共 50 个units 
fc_w1 = tf.Variable(tf.random_normal([2*2*5,50]))
fc_b1 = tf.Variable(tf.random_normal([50]))
fc_out1 = tf.nn.relu(tf.matmul(max_pool2_flat,fc_w1) + fc_b1)

# 输出层
out_w1 = tf.Variable(tf.random_normal([50,10]))
out_b1 = tf.Variable(tf.random_normal([10]))
pred = tf.nn.softmax(tf.matmul(fc_out1,out_w1) + out_b1)
print(pred)

# 定义 loss 和 optimizer
# 看做定义连接关系 仍未执行

# reduce_mean 在tensor的某一维度上求平均值的函数, 如果不指定维数，那么就在所有的元素中取平均值
# tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。
loss = -tf.reduce_mean(tf_Y * tf.log(tf.clip_by_value(pred,1e-11,1.0)))
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
# tf.argmax就是返回最大的那个数值所在的下标
# axis = 1 比较范围缩小了，只会比较每个数组内的数的大小
y_pred = tf.argmax(pred,1)
print (y_pred)
bool_pred = tf.equal(tf.argmax(tf_Y,1),y_pred)

#tf.cast(x, dtype, name=None) 此函数是类型转换函数
accuracy = tf.reduce_mean(tf.cast(bool_pred,tf.float32))

with tf.Session() as sess:
    #initialize_all_variables已被弃用，使用tf.global_variables_initializer代替
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        # 每个周期进行MBGD算法
        for batch_xs,batch_ys in generate_batch(X,Y,Y.shape[0],batch_size):
            sess.run(train_step,feed_dict = {tf_X:batch_xs,tf_Y:batch_ys})
        if epoch %100 == 0:
            # training accucary 
            res = sess.run(accuracy,feed_dict = {tf_X:X,tf_Y:Y})
            print (epoch,res)
    # 在使用t.eval()时，等价于：tf.get_default_session().run(t)
    # 最主要的区别就在于你可以使用sess.run()在同一步获取多个tensor中的值
    #tu.eval()  # runs one step
    #ut.eval()  # runs one step
    #sess.run([tu, ut])  # evaluates both tensors in a single step
    res_ypred = y_pred.eval(feed_dict={tf_X:X,tf_Y:Y}).flatten() 
    print (res_ypred)
    
#这个模型还不能用来预测单个样本
#因为在进行BN层计算时，单个样本的均值和方差都为0，会得到相反的预测效果，解决方法详见归一化层
#第100次个batch size 迭代时，准确率就快速接近收敛了，这得归功于Batch Normalization 的作用
from sklearn.metrics import  accuracy_score
print (accuracy_score(Y_data,res_ypred.reshape(-1,1)))
