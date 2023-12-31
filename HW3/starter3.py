from HW3.src.model import simple_ffnn, deep_ffnn
from HW3.src.model.metrics.q1 import generate_metrics_for_q1
from HW3.src.model.metrics.q2 import generate_metrics_for_q2
from HW3.src.model.metrics.q3 import generate_metrics_for_q3
from HW3.src.model.metrics.q4 import generate_metrics_for_q4


def read_mnist(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show_mnist(file_name,mode):
    
    data_set = read_mnist(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
                   
def read_insurability(file_name):
    
    count = 0
    data = []
    with open(file_name,'rt') as f:
        for line in f:
            if count > 0:
                line = line.replace('\n','')
                tokens = line.split(',')
                if len(line) > 10:
                    x1 = float(tokens[0])
                    x2 = float(tokens[1])
                    x3 = float(tokens[2])
                    if tokens[3] == 'Good':
                        cls = 0
                    elif tokens[3] == 'Neutral':
                        cls = 1
                    else:
                        cls = 2
                    data.append([[cls],[x1,x2,x3]])
            count = count + 1
    return(data)


def classify_insurability():
    
    train = read_insurability('src/data/three_train.csv')
    valid = read_insurability('src/data/three_valid.csv')
    test = read_insurability('src/data/three_test.csv')

    # insert code to train simple FFNN and produce evaluation metrics

    # train
    parameters = {
        "input_size": 3,
        "hidden_size": 2,
        "output_size": 3,
        "learning_rate": 0.001,
        "bias": True,
        "epochs": 1000,
    }

    simple_ffnn.train(
        parameters,
        train,
        valid,
        "insurability",
        False,
        "./src/model/trained/q1"
    )

    generate_metrics_for_q1(test)


def classify_mnist():
    
    train = read_mnist('src/data/mnist_train.csv')
    valid = read_mnist('src/data/mnist_valid.csv')
    test = read_mnist('src/data/mnist_test.csv')
    # show_mnist('src/data/mnist_test.csv', 'pixels')
    
    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics

    # train
    parameters = {
        "input_size": 784,
        "hidden_size": 397,
        "output_size": 10,
        "epochs": 1000,
        "learning_rate": 0.001,
        "weight_decay": 0,
    }

    deep_ffnn.train(
        parameters,
        train,
        valid,
        "mnist",
        "./src/model/trained/q2"
    )

    generate_metrics_for_q2(test)


def classify_mnist_reg():
    
    train = read_mnist('src/data/mnist_train.csv')
    valid = read_mnist('src/data/mnist_valid.csv')
    test = read_mnist('src/data/mnist_test.csv')
    show_mnist('src/data/mnist_test.csv', 'pixels')
    
    # add a regularizer of your choice to classify_mnist()

    # train
    parameters = {
        "input_size": 784,
        "hidden_size": 100,
        "output_size": 10,
        "epochs": 1000,
        "learning_rate": 0.001,
        "weight_decay": 0.1,
    }

    deep_ffnn.train(
        parameters,
        train,
        valid,
        "mnist",
        "./src/model/trained/q3"
    )

    generate_metrics_for_q3(test)

def classify_insurability_manual():
    
    train = read_insurability('src/data/three_train.csv')
    valid = read_insurability('src/data/three_valid.csv')
    test = read_insurability('src/data/three_test.csv')
    
    # reimplement classify_insurability() without using a PyTorch optimizer.
    # this part may be simpler without using a class for the FFNN

    # train
    parameters = {
        "input_size": 3,
        "hidden_size": 2,
        "output_size": 3,
        "learning_rate": 0.001,
        "bias": False,
        "epochs": 1000,
    }

    simple_ffnn.train(
        parameters,
        train,
        valid,
        "insurability",
        True,
        "./src/model/trained/q4"
    )

    generate_metrics_for_q4(test)
    
    
def main():
    classify_insurability()
    classify_mnist()
    classify_mnist_reg()
    classify_insurability_manual()


if __name__ == "__main__":
    main()
