import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
mnist=input_data.read_data_sets("J:\Softwares & Programming Languages\TensorFlow\MNIST/", one_hot=True)
#setting up no of nodes and layers
n_nodes_hl1=1000
n_nodes_hl2=1000
n_nodes_hl3=1000
n_nodes_hl4=1000
n_nodes_hl5=1000

n_classes=10
batch_size=100

x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')

# Setting up variables for hidden layers
def neural_network_model(data):
    hidden_1_layer={'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_3_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    hidden_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}
    
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}
    #modelling a neural network= (input*weight)+bias
    
    l1=tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
    #threshold function(rectify linear?)
    l1=tf.nn.relu(l1)
    
    l2=tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2=tf.nn.relu(l2)
    
    l3=tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    l3=tf.nn.relu(l3)
    
    l4 = tf.add(tf.matmul(l3,hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)
    
    output = tf.matmul(l4,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost= tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer=tf.train.AdamOptimizer().minimize(cost)
    #epoch= forward + back propagation
    hm_epochs=10
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(hm_epochs):
            epoch_loss=0
            
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y=mnist.train.next_batch(batch_size)
                # optimizing cost with x and y 
                _,c=sess.run([optimizer,cost], feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss +=c
            print('Epoch',epoch, ' completed out of', hm_epochs,'loss:', epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

train_neural_network(x)

