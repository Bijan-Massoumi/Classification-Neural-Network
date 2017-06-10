import classification_net as cn
from activation_methods import *
import preprocess as p


if __name__ == "__main__":
    train,target = p.load_images('train.csv')
    net = cn.ClassificationNetwork([train.shape[1],25,10],Activations.sigmoid)
    image = train[0]
    print(net.propagate_forward(image)[-1])
