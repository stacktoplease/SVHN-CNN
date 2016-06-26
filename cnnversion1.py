import os.path
import urllib
import numpy as np
import scipy.io as scp

testfile = urllib.URLopener()
testfile2=urllib.URLopener()

if not  os.path.isfile("test.mat"):
    testfile.retrieve("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", "test.mat")

if not  os.path.isfile("train.mat"):
    testfile.retrieve("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", "train.mat")
testdata=scp.loadmat('test.mat')
traindata=scp.loadmat('train.mat')

trainDataX = traindata['X'].astype('float32') /256                                                                                                              
testDataX = testdata['X'].astype('float32') / 256      

trainDataY = traindata['y']
testDataY = testdata['y']
def OnehotEncoding(Y):
    Ytr=[]
    for el in Y:
        temp=np.zeros(10)
        if el==10:
            temp[0]=1
        elif el==1:
            temp[1]=1
        elif el==2:
            temp[2]=1
        elif el==3:
            temp[3]=1
        elif el==4:
            temp[4]=1
        elif el==5:
            temp[5]=1
        elif el==6:
            temp[6]=1
        elif el==7:
            temp[7]=1
        elif el==8:
            temp[8]=1
        elif el==9:
            temp[9]=1
        Ytr.append(temp)
    return np.asarray(Ytr)


trainDataY = OnehotEncoding(trainDataY)
testDataY = OnehotEncoding(testDataY)
def transposeArray(data):
    print 'started'
    xtrain = []
    trainLen = data.shape[3]
    print trainLen
    for x in xrange(trainLen):
        xtrain.append(data[:,:,:,x])

    xtrain = np.asarray(xtrain)
    return xtrain
trainDataX = transposeArray(trainDataX)
testDataX = transposeArray(testDataX)
print trainDataX.shape
import tensorflow as tf
sess = tf.InteractiveSession()


x = tf.placeholder(tf.float32, shape=[None, 32,32,3])

y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,32,32,3])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
W_fc1 = weight_variable([8 * 8* 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
#y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv = tf.nn.softmax(logits)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print logits
with tf.Session() as sess:
    #epoch=20000
    epoch=300000
    batch_size=100
    sess.run(tf.initialize_all_variables())
    p = np.random.permutation(range(len(trainDataX)))
    trX, trY = trainDataX[p], trainDataY[p]
    print len(trainDataX)
    start = 0
    end = 0  
    for step in range(epoch):
        start = end
        end = start + batch_size

        if start >= len(trainDataX):
            start = 0
            end = start + batch_size

        if end >= len(trainDataX):
            end = len(trainDataX) - 1
        if start == end:
            start = 0
            end = start + batch_size
        inX, outY = trX[start:end], trY[start:end]
        #sess.run(optimizer, feed_dict= {x: inX, y_: outY, keep_prob:0.75})
        train_step.run(feed_dict={x: inX, y_: outY, keep_prob: 0.5})
        if step % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: inX, y_: outY, keep_prob:1})
            print 'cost at each step :', step, 'is :', sess.run(cross_entropy, feed_dict={x: inX, y_: outY, keep_prob:1.0})
            print("step %d, training accuracy %g"%(step, train_accuracy))
    print len(testDataY)
    list_testbatch_accuracy=[]
    counter=len(testDataY)
    start=0
    end=0
    testbatch=1
    while end<counter-testbatch:
        start=end
        end+=testbatch
        list_testbatch_accuracy.append(accuracy.eval(feed_dict={
    x: testDataX[start:end], y_:testDataY[start:end] , keep_prob: 1.0}))
        #print "start" ,start
        #print "end" ,end
    start=end
    end+=counter%testbatch
    list_testbatch_accuracy.append(accuracy.eval(feed_dict={
    x: testDataX[start:end], y_:testDataY[start:end] , keep_prob: 1.0}))
    #print "start" ,start
    #print "end" ,end
    print np.mean(list_testbatch_accuracy)