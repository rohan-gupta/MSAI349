import matplotlib.pyplot as plt
import unit_tests, ID3, parse, random, utils


def get_avg_of_100(inFile):
    set_of_size = range(10,300) #x-axis
    acc_of_training = []
    acc_of_validation = []

    data = parse.parse(inFile)
    data = utils.update_missing_attributes_with_majority_value(data)
    
    for setSize in set_of_size:
        random.shuffle(data)
        avg_train , avg_validation = cal_avg_of_100(data[:setSize])
        print(setSize, avg_train, '\n\naas')
        acc_of_training.append(avg_train)
        acc_of_validation.append(avg_validation)
    
    plt.figure(figsize=(8, 5))
    plt.plot(set_of_size, acc_of_training, label='Training Accuracy')
    plt.plot(set_of_size, acc_of_validation, label='Validation Accuracy')
    # plt.plot(size_of_set, test_accuracy, label='Testing Accuracy')
    plt.xlabel('size_of_set')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy vs. size_of_set')
    plt.legend()
    plt.grid(True)
    plt.show()
  






def cal_avg_of_100(data):

    withPruning = []
    withoutPruning = []

    training_100 = []
    validation_100 = []
    train = data[:len(data)//2]
    valid = data[len(data)//2:3*len(data)//4]
    test = data[3*len(data)//4:]

    for i in range(100):
        print(len(train), len(valid), len(test))
        random.shuffle(train)
        random.shuffle(valid)
        random.shuffle(test)
        
    
        tree = ID3.ID3(train, 'democrat')
        acc = ID3.test(tree, train)
        training_100.append(acc)
        acc = ID3.test(tree, valid)
        # print("validation accuracy: ",acc)
        validation_100.append(acc)
        acc = ID3.test(tree, test)
        # print("test accuracy: ",acc)
    
        # ID3.prune(tree, valid)
        # acc = ID3.test(tree, train)
        # print("pruned tree train accuracy: ",acc)
        # acc = ID3.test(tree, valid)
        # print("pruned tree validation accuracy: ",acc)
        # acc = ID3.test(tree, test)
        # print("pruned tree test accuracy: ",acc)
        # withPruning.append(acc)
        # tree = ID3.ID3(train+valid, 'democrat')
        # acc = ID3.test(tree, test)
        # print("no pruning test accuracy: ",acc)
        # withoutPruning.append(acc)
    # print(withPruning)
    # print(withoutPruning)
    # print("average with pruning",sum(withPruning)/len(withPruning)," without: ",sum(withoutPruning)/len(withoutPruning))
    
    return sum(training_100)/len(training_100) , sum(validation_100)/len(validation_100)

if __name__ == "__main__":
  get_avg_of_100("house_votes_84.data")
