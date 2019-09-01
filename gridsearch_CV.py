# artificial neural networks

import pandas as pd
data = pd.read_csv('Churn_Modelling.csv')


x = data.iloc[:,3:-1].values
y = data.iloc[:,-1].values

# encode
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lx1 = LabelEncoder()
x[:,1] = lx1.fit_transform(x[:,1])
lx2 = LabelEncoder()
x[:,2] = lx2.fit_transform(x[:,2])
one = OneHotEncoder(categorical_features=[1])
x = one.fit_transform(x).toarray()

# split
from sklearn.model_selection import train_test_split
x_train , x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 )

#Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test= sc.transform(x_test)

# making ann
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer) :
    cl = Sequential()
    cl.add(Dense(9 ,kernel_initializer ='uniform',activation='relu',input_dim = 12))
    cl.add(Dense(6 ,kernel_initializer ='uniform',activation='relu'))
    cl.add(Dense(1 ,kernel_initializer ='uniform',activation='sigmoid'))
    cl.compile(optimizer = optimizer ,loss='binary_crossentropy',metrics=['accuracy'])
    return cl


cl = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size':[25,35] , 'epochs' : [15] , 'optimizer' : ['adam' , 'adamax']}

grid_search = GridSearchCV(estimator = cl , param_grid = parameters , scoring = 'accuracy',cv=5)
# in cv method it will create 10 pieces. these pieces will be compiled as 9 training and 1 tetsing


grid_search = grid_search.fit(x_train,y_train)


best_param = grid_search.best_params_
best_acc = grid_search.best_score_



















