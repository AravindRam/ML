import sys
import math

#Function to compute the parent and child entropy

def compute_entropy(dict1,dict2):
    for j in range(1,No_of_features):
        parent_entropy = 0.0
        feature_value_list = unique_attr[j-1]
        for feature_value in feature_value_list:
            child_entropy = 0.0
            for label in unique_actual_labels:     #Retrieving values of p and e and computing the child entropy using the formula
                pvalue = dict1[j][feature_value]['p']
                evalue = dict1[j][feature_value]['e']
                numerator = dict1[j][feature_value][label]
                if(numerator/(pvalue+evalue) != 0.0):
                    child_entropy+= (- numerator/(pvalue+evalue)) * math.log(numerator/(pvalue+evalue),2)
                else:
                    child_entropy+=0.0
            dict1[j][feature_value]['entropy']=child_entropy
        for label in unique_actual_labels:         #Retrieving values of p and e and computing the parent entropy using the formula
            ppvalue = dict2[j]['p']
            pevalue = dict2[j]['e']
            pnumerator = dict2[j][label]
            if(pnumerator/(ppvalue+pevalue)):
                parent_entropy+= (- pnumerator/(ppvalue+pevalue)) * math.log(pnumerator/(ppvalue+pevalue),2)
            else:
                parent_entropy+=0.0
        dict2[j]['entropy']=parent_entropy
    return dict1,dict2

# Function to compute information gain

def compute_info_gain(dict1,dict2):
    gain = []
    for j in range(1,No_of_features):
        feature_value_list = unique_attr[j-1]       #Get the unique values for each feature
        total=0.0
        ppvalue = dict2[j]['p']                     #Retrieve the p and e values of the parent
        pevalue = dict2[j]['e']
        for feature_value in feature_value_list:
            pvalue = dict1[j][feature_value]['p']   #Retrieve the p and e values of each child
            evalue = dict1[j][feature_value]['e']
            child_entropy = dict1[j][feature_value]['entropy']
            total+=(pvalue+evalue)/(ppvalue+pevalue)*child_entropy      #Find the weighted sum of child entropy
        parent_entropy = dict2[j]['entropy']                        #Retrieve the parent entropy
        info_gain = parent_entropy-total            #Compute the info gain using the formula
        gain.append(info_gain)
    return gain.index(max(gain))                    #return the index of the feature with max info gain




T = int(sys.argv[1])	#No of boosting iterations
train_file = sys.argv[2]	#Training File
test_file = sys.argv[3]		#Test File

#Reading the training file data into a list of lists

data = []
train = open(train_file,"r")
i=0
for lines in train.readlines():
    line = lines.split("\r")
    for word in line:
        tokens = word.split("\t")
        data.append(tokens)

#Storing all the actual labels in a list

actual_labels=[]
for i in range(len(data)):
    actual_labels.append(data[i][0])

#Storing all the features/attributes into an array of lists starting from attr[0] for feature 1 to attr[20] for feature 21

No_of_features = len(data[0])
attr = [0]*No_of_features
for j in range(1,No_of_features):
    attr[j]=[]
    for i in range(len(data)):
        attr[j].append(data[i][j])
del attr[0]

#Finding the unique list of actual labels

unique_actual_labels = list(set(actual_labels))

#Finding the unique list of attributes/values for each feature

unique_attr=[[]]
for j in range(No_of_features-1):
    unique_attr.append(list(set(attr[j])))
del unique_attr[0]

#Initializing the weights to be 1/m for each of the m training examples

weights = [1/float(len(data))]*len(data)

t=1
final_classifier={}
final_alpha=[]
final_attributes=[]

#Run a loop for the number of iterations passed as input via the command line argument

while(t<=T):
    CountDict={}                        #Dictionary which contains p,e and entropy for each child/attribute values
    for j in range(1,No_of_features):
        CountDict[j]={}
        for i in range(len(data)):
            CountDict[j][data[i][j]]={}
            for label in unique_actual_labels:          #Initializing the dictionary
                CountDict[j][data[i][j]].update({label:0})
                CountDict[j][data[i][j]].update({"entropy":0})

    for j in range(1,No_of_features):                   #Adding the weights for each feature for all the examples
        for i in range(len(data)):
            CountDict[j][data[i][j]][data[i][0]]+=weights[i]

    instances={}                        #Dictionary which contains p,e and entropy for each parent/feature
    for j in range(No_of_features-1):
        instances[j+1]={}
        feature_value_list = unique_attr[j]
        for label in unique_actual_labels:
            total=0
            for feature_value in feature_value_list:
                total+= float(CountDict[j+1][feature_value][label])
            instances[j+1][label]=total         #Weight of each label in the parent will be the sum of all the weights in the child
            instances[j+1].update({"entropy":0})

    CountDict,instances = compute_entropy(CountDict,instances)   #Call the method to compute the parent and child entropy
    best_attribute = compute_info_gain(CountDict,instances)      #Call the method to compute the info gain and get the best attribute

    classifier={}
    classifier[best_attribute+1]={}
    feature_value_list = unique_attr[best_attribute]
    predicted_labels=['p']*len(data)

    #Calculate the predicted labels for the best attribute selected

    for feature_value in feature_value_list:
        classifier[best_attribute+1][feature_value]={}
        if(CountDict[best_attribute+1][feature_value]['p'] > CountDict[best_attribute+1][feature_value]['e']):
            assign='p'
            classifier[best_attribute+1][feature_value].update({"class":-1})
        else:
            assign='e'
            classifier[best_attribute+1][feature_value].update({"class":1})
        indices=[]
        for i in range(len(data)):
            if(data[i][best_attribute+1] == feature_value):
                indices.append(i)
        for i in indices:
            predicted_labels[i]=assign

    final_attributes.append(best_attribute+1)
    final_classifier.update({t:classifier})
    actual_labels_mapped=[0]*len(data)
    predicted_labels_mapped=[0]*len(data)

    #Convert all the labels by mapping e to 1 and p to -1 for actual and predicted labels

    for i in range(len(data)):
        if(actual_labels[i]=='p'):
            actual_labels_mapped[i]=-1
        elif(actual_labels[i]=='e'):
            actual_labels_mapped[i]=1
        if(predicted_labels[i]=='p'):
            predicted_labels_mapped[i]=-1
        elif(predicted_labels[i]=='e'):
            predicted_labels_mapped[i]=1

    #Compute a list of misclassified labels by appending 0 if it is misclassified and 1 if it is correctly predicted

    misclassified = []
    for i in range(len(data)):
        if(actual_labels[i] == predicted_labels[i]):
            misclassified.append(1)
        else:
            misclassified.append(0)

    #Compute the training error, epsilon by adding the weights if the labels are misclassified.

    epsilon=0
    for i in range(len(data)):
        if(actual_labels_mapped[i]!=predicted_labels_mapped[i]):
            epsilon+=weights[i]

    #Compute the strength of the classifier, alpha and append it to a list to store the alpha for each iteration

    alpha=0.5*math.log((1-epsilon)/epsilon,math.e)
    final_alpha.append(alpha)

    #Update the weights for each iteration

    total_weight=0
    for i in range(len(data)):
        total_weight+=weights[i]*math.exp(-(alpha*actual_labels_mapped[i]*predicted_labels_mapped[i]))

    for i in range(len(data)):
        weights[i]=(weights[i]*math.exp(-(alpha*actual_labels_mapped[i]*predicted_labels_mapped[i])))/total_weight

    t=t+1

#Testing

#Reading the test file data into a list of lists

test_data = []
test = open(test_file,"r")
i=0
for lines in test.readlines():
    line = lines.split("\r")
    for word in line:
        tokens = word.split("\t")
        test_data.append(tokens)

#Read each data from the test file and use the trained classifer to retrieve the predicted class for the selected features

misclassified_test=[]
for i in range(len(test_data)):
    finalValue=0.0
    for t in range(1,T+1):
        if(test_data[i][final_attributes[t-1]] in final_classifier[t][final_attributes[t-1]].keys()):   #Check if data/attribute value is present in the classifier
            finalValue+=final_alpha[t-1]*final_classifier[t][final_attributes[t-1]][test_data[i][final_attributes[t-1]]]['class']
        else:
            finalValue+=final_alpha[t-1]*1      #If the data/attribute value is not present, predict the class to be e or 1
    if(finalValue>=0):
        predicted_label='e'             #Predict the label to be e if finalValue >=0 and p if finalValue < 0
    elif(finalValue<0):
        predicted_label='p'

    if(test_data[i][0] == predicted_label): #Append 1 to list if labels are misclassified and 0 if labels are predicted correctly
        misclassified_test.append(0)
    else:
        misclassified_test.append(1)

#Compute the test accuracy based on the number of correctly classified labels on the test data by the classifier

test_accuracy=(len(test_data)-sum(misclassified_test))/float(len(test_data))*100

#Print the test accuracy and list of alpha values computed at each iteration
print test_accuracy
for a in final_alpha:
    print a

train.close()
test.close()