import classification_net as cn
from activation_methods import *
import preprocess as p


if __name__ == "__main__":
    train,target = p.load_images('train.csv')
    train, target, test, target_test = p.partition_set(.8,train,target)
    net = cn.ClassificationNetwork([train.shape[1],25,10],Activations.sigmoid)
    net.train_network(train,target,epochs=10,reg_strength=0.0001,learning_rate=0.0001,batch_size=1000,momentum=0.95,debug = False)
    print(net.compute_accuracy(test,target_test))
