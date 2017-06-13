import classification_net as cn
from activation_methods import *
import preprocess as p


if __name__ == "__main__":
    train,target = p.load_images('train.csv')
    train, target, test, target_test = p.partition_set(.8,train,target)
    net = cn.ClassificationNetwork([train.shape[1],128,64,10],Activations.sigmoid)
    net.train_network(train,target,epochs=100,reg_strength=0.001,learning_rate=0.00001,batch_size=100,momentum=0.95,debug = False)
    print(net.compute_accuracy(test,target_test))
