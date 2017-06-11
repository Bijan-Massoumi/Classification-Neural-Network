import classification_net as cn
from activation_methods import *
import preprocess as p


if __name__ == "__main__":
    train,target = p.load_images('train.csv')
    test,target_test = p.load_images('test.csv')
    net = cn.ClassificationNetwork([train.shape[1],25,10],Activations.sigmoid)
    net.train_network(train,target)
    print(net.compute_accuracy(test,target_test))