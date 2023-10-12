import matplotlib.pyplot as plt
import unit_tests, ID3, parse, random, utils


def get_avg_of_100(inFile):
    set_of_size = range(10,300) #x-axis
    acc_of_training = []
    acc_of_validation = []
    acc_of_test = []
    p_acc_of_training = []
    p_acc_of_validation = []
    p_acc_of_test = []


    data = parse.parse(inFile)
    data = utils.update_missing_attributes_with_majority_value(data)
    
    for setSize in set_of_size:
        print(setSize)
        random.shuffle(data)
        avg_train , avg_validation, avg_test, p_avg_train , p_avg_validation, p_avg_test  = cal_avg_of_100(data[:setSize])
        acc_of_training.append(avg_train)
        acc_of_validation.append(avg_validation)
        acc_of_test.append(avg_test)

        p_acc_of_training.append(p_avg_train)
        p_acc_of_validation.append(p_avg_validation)
        p_acc_of_test.append(p_avg_test)
        
    
    plt.figure(figsize=(8, 5))
    # plt.plot(set_of_size, acc_of_training, label='Training Accuracy')
    # plt.plot(set_of_size, acc_of_validation, label='Validation Accuracy')
    # plt.plot(set_of_size, p_acc_of_training, label='P_Training Accuracy')
    # plt.plot(set_of_size, p_acc_of_validation, label='P_Validation Accuracy')
    plt.plot(set_of_size, acc_of_test, label='Testing Accuracy Without Pruning') #Plot the testing result
    plt.plot(set_of_size, p_acc_of_test, label='Testing Accuracy With Pruning') #Plot the testing result

    plt.xlabel('Size of Set')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy vs. Size Of Set')
    plt.legend()
    plt.grid(True)
    plt.show()
  


def cal_avg_of_100(data):

    withPruning = []
    withoutPruning = []

    training_100 = []
    validation_100 = []
    pruning_training_100 = []
    pruning_validation_100 = []

    train_val = data[:len(data)//4]
    test = data[3*len(data)//4:]

    for i in range(100):
        random.shuffle(train_val)
        train = train_val[:2*len(train_val)//3]
        valid = train_val[2*len(train_val)//3 :] 
        random.shuffle(test)
        
    
        tree = ID3.ID3(train, 'democrat')
        acc = ID3.test(tree, train)
        training_100.append(acc)
        acc = ID3.test(tree, valid)
        # print("validation accuracy: ",acc)
        validation_100.append(acc)
        acc = ID3.test(tree, test)
        # print("test accuracy: ",acc)
    
        ID3.prune(tree, valid)
        acc = ID3.test(tree, train)
        # print("pruned tree train accuracy: ",acc)
        pruning_training_100.append(acc)
        acc = ID3.test(tree, valid)
        # print("pruned tree validation accuracy: ",acc)
        pruning_validation_100.append(acc)
        acc = ID3.test(tree, test)
        # print("pruned tree test accuracy: ",acc)
        withPruning.append(acc)
        tree = ID3.ID3(train+valid, 'democrat')
        acc = ID3.test(tree, test)
        # print("no pruning test accuracy: ",acc)
        withoutPruning.append(acc)
    # print(withPruning)
    # print(withoutPruning)
    print("average with pruning",sum(withPruning)/len(withPruning)," without: ",sum(withoutPruning)/len(withoutPruning))
    
    return sum(training_100)/len(training_100) , sum(validation_100)/len(validation_100) ,sum(withoutPruning)/len(withoutPruning), sum(pruning_training_100)/len(pruning_training_100) , sum(pruning_validation_100)/len(pruning_validation_100), sum(withPruning)/len(withPruning)

if __name__ == "__main__":
  get_avg_of_100("house_votes_84.data")
