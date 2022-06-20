# Titanic Survivor Classification

## Description
This is notebook to build classification for Titanic passengers to classify if those passengers survived or not. Our aim is to get model with the most accuracy possible.

source : https://www.kaggle.com/competitions/titanic/data

## Data Overview


```python
import pandas as pd
data = pd.read_csv('train.csv')
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    


```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.nunique()
```




    PassengerId    891
    Survived         2
    Pclass           3
    Name           891
    Sex              2
    Age             88
    SibSp            7
    Parch            7
    Ticket         681
    Fare           248
    Cabin          147
    Embarked         3
    dtype: int64



So, here is Describe for each column
- PassengerId : id for each passenger for this data
- Survived : if this passenger survived or not (0 = No, 1 = Yes)
- Pclass : Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- Name : name of passenger
- Sex : sex of passenger
- Age : age of passenger
- SibSp : number of siblings / spouses aboard the Titanic
- Parch : number of parents / children aboard the Titanic
- Ticket : Ticket number
- Fare : Passenger fare
- Cabin : Cabin number
- Embarked : Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Clean Up
Remove column that we are not going to use and fill null data.


```python
# remove cloumns PassengerId, Name, Ticket and Cabin Since it have high cardinality.
data = data[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
```


```python
# get mean age
mean_age = data['Age'].mean()

print('mean age = ', mean_age)

# show histogram for Embarked for impute with the most frequent value
mode_emb = data['Embarked'].mode()
print('mode Embarked = ', mode_emb[0])
```

    mean age =  29.69911764705882
    mode Embarked =  S
    


```python
# impute missing data
values = {'Age' : mean_age, 'Embarked': 'S'}
data = data.fillna(value=values)
```


```python
# convert Sex to int and Embarked to 3 int columns
data['Sex'] = data.Sex.apply(lambda x: int(x =='male'))
data['Emb_C'] = data.Embarked.apply(lambda x: int(x =='C'))
data['Emb_Q'] = data.Embarked.apply(lambda x: int(x =='Q'))
data['Emb_S'] = data.Embarked.apply(lambda x: int(x =='S'))

data = data[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Emb_C', 'Emb_Q', 'Emb_S']]
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Emb_C</th>
      <th>Emb_Q</th>
      <th>Emb_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Visual Display


```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(data, kind="hist")
plt.savefig('pair_plot1.png')
```


    
![png](output_14_0.png)
    



```python
corr_data = data.corr()
print(corr_data)
```

              Survived    Pclass       Sex       Age     SibSp     Parch  \
    Survived  1.000000 -0.338481 -0.543351 -0.069809 -0.035322  0.081629   
    Pclass   -0.338481  1.000000  0.131900 -0.331339  0.083081  0.018443   
    Sex      -0.543351  0.131900  1.000000  0.084153 -0.114631 -0.245489   
    Age      -0.069809 -0.331339  0.084153  1.000000 -0.232625 -0.179191   
    SibSp    -0.035322  0.083081 -0.114631 -0.232625  1.000000  0.414838   
    Parch     0.081629  0.018443 -0.245489 -0.179191  0.414838  1.000000   
    Fare      0.257307 -0.549500 -0.182333  0.091566  0.159651  0.216225   
    Emb_C     0.168240 -0.243292 -0.082853  0.032024 -0.059528 -0.011069   
    Emb_Q     0.003650  0.221009 -0.074115 -0.013855 -0.026354 -0.081228   
    Emb_S    -0.149683  0.074053  0.119224 -0.019336  0.068734  0.060814   
    
                  Fare     Emb_C     Emb_Q     Emb_S  
    Survived  0.257307  0.168240  0.003650 -0.149683  
    Pclass   -0.549500 -0.243292  0.221009  0.074053  
    Sex      -0.182333 -0.082853 -0.074115  0.119224  
    Age       0.091566  0.032024 -0.013855 -0.019336  
    SibSp     0.159651 -0.059528 -0.026354  0.068734  
    Parch     0.216225 -0.011069 -0.081228  0.060814  
    Fare      1.000000  0.269335 -0.117216 -0.162184  
    Emb_C     0.269335  1.000000 -0.148258 -0.782742  
    Emb_Q    -0.117216 -0.148258  1.000000 -0.499421  
    Emb_S    -0.162184 -0.782742 -0.499421  1.000000  
    


```python
sns.heatmap(corr_data)
```




    <AxesSubplot:>




    
![png](output_16_1.png)
    



```python
plot_data = data[['Age','SibSp','Parch','Fare']]
plot_data.plot(kind="box", subplots=True, figsize =(10, 5))
```




    Age         AxesSubplot(0.125,0.11;0.168478x0.77)
    SibSp    AxesSubplot(0.327174,0.11;0.168478x0.77)
    Parch    AxesSubplot(0.529348,0.11;0.168478x0.77)
    Fare     AxesSubplot(0.731522,0.11;0.168478x0.77)
    dtype: object




    
![png](output_17_1.png)
    


### Analysis
From information above, seems there is no strong correlation between features as you and see from correlation heatmap or raw data (>0.7). You may see Emb_C and Emb_S have high value but those columns was extracted from Embarked feature. There are lot of outiler data as you can see. But, I think those are all valid. Age are in reasonable range (more than 0 and less than 90). SibSp and Parch also resonable because most of people board alone. Fare also reasonbale for me.

## Classification
I want to use Random Forest Tree to solve this problem. Because we have only fews features.

First, split data in to train and test (test for 20%).


```python
from sklearn.model_selection import train_test_split

X = data[['Pclass','Sex','Age','SibSp','Parch','Fare','Emb_C', 'Emb_Q', 'Emb_S']]
Y = data['Survived']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1992)

print(len(x_train), len(x_test))
```

    712 179
    

Then, Build first model to be our base line.


```python
from sklearn.ensemble import RandomForestClassifier

bl_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=1992).fit(x_train, y_train)
print('score with training data = ',bl_model.score(x_train,y_train))
print('score with test data = ',bl_model.score(x_test,y_test))
```

    score with training data =  0.8539325842696629
    score with test data =  0.7653631284916201
    

try to play with n_estimators and max_depth


```python
es = list(range(10,110,10))
train_scores = []
test_scores = []
for e in es:
    model = RandomForestClassifier(n_estimators=e, max_depth=5, random_state=1992).fit(x_train, y_train)
    train_scores.append(model.score(x_train,y_train))
    test_scores.append(model.score(x_test,y_test))
    
print('train_scores = ', train_scores)
print('test_scores = ', test_scores)

plt.plot(es, train_scores)
plt.plot(es, test_scores)

plt.legend(['train_scores', 'test_scores'])
plt.show()
```

    train_scores =  [0.8539325842696629, 0.8609550561797753, 0.8651685393258427, 0.8679775280898876, 0.8567415730337079, 0.8567415730337079, 0.8567415730337079, 0.8567415730337079, 0.8553370786516854, 0.8553370786516854]
    test_scores =  [0.7653631284916201, 0.7821229050279329, 0.7932960893854749, 0.7932960893854749, 0.7988826815642458, 0.8100558659217877, 0.8100558659217877, 0.8100558659217877, 0.8100558659217877, 0.8100558659217877]
    


    
![png](output_25_1.png)
    


Try with max_depth=10


```python
es = list(range(10,110,10))
train_scores = []
test_scores = []
for e in es:
    model = RandomForestClassifier(n_estimators=e, max_depth=10, random_state=1992).fit(x_train, y_train)
    train_scores.append(model.score(x_train,y_train))
    test_scores.append(model.score(x_test,y_test))
    
print('train_scores = ', train_scores)
print('test_scores = ', test_scores)

plt.plot(es, train_scores)
plt.plot(es, test_scores)

plt.legend(['train_scores', 'test_scores'])
plt.show()
```

    train_scores =  [0.9424157303370787, 0.9466292134831461, 0.9480337078651685, 0.9508426966292135, 0.949438202247191, 0.952247191011236, 0.9550561797752809, 0.9536516853932584, 0.949438202247191, 0.9508426966292135]
    test_scores =  [0.7932960893854749, 0.8100558659217877, 0.8156424581005587, 0.8156424581005587, 0.8268156424581006, 0.8212290502793296, 0.8324022346368715, 0.8324022346368715, 0.8268156424581006, 0.8324022346368715]
    


    
![png](output_27_1.png)
    


From graphs we can see that using max_depth=10 is better. And the reasonal n_estimators is 50.

I will try to tune ccp_alpha to reduce data overfitted.


```python
alphas = []
for i in range(10):
    alphas.append(i*0.005)

train_scores = []
test_scores = []
for a in alphas:
    model = RandomForestClassifier(n_estimators=50, max_depth=10, ccp_alpha=a, random_state=1992).fit(x_train, y_train)
    train_scores.append(model.score(x_train,y_train))
    test_scores.append(model.score(x_test,y_test))
    
print('train_scores = ', train_scores)
print('test_scores = ', test_scores)

plt.plot(alphas, train_scores)
plt.plot(alphas, test_scores)

plt.legend(['train_scores', 'test_scores'])
plt.show()
```

    train_scores =  [0.949438202247191, 0.8595505617977528, 0.8356741573033708, 0.8146067415730337, 0.8188202247191011, 0.7921348314606742, 0.7879213483146067, 0.7879213483146067, 0.7879213483146067, 0.7879213483146067]
    test_scores =  [0.8268156424581006, 0.7988826815642458, 0.7988826815642458, 0.7877094972067039, 0.770949720670391, 0.7821229050279329, 0.7821229050279329, 0.7821229050279329, 0.7821229050279329, 0.7821229050279329]
    


    
![png](output_30_1.png)
    


Incrase ccp_alpha helps reduce over fitted problem. But, I don't want to sacrifice score fro this.

So, the best model for me is n_estimators=50, max_depth=10 and ccp_alpha=0


```python
best_model = RandomForestClassifier(n_estimators=50, max_depth=10, ccp_alpha=0, random_state=1992).fit(x_train, y_train)
print('score with training data = ',bl_model.score(x_train,y_train))
print('score with test data = ',bl_model.score(x_test,y_test))
```

    score with training data =  0.8539325842696629
    score with test data =  0.7653631284916201
    

Use this setting and train with all data we have to predict test data from Kaggle.


```python
best_model = RandomForestClassifier(n_estimators=50, max_depth=10, ccp_alpha=0, random_state=1992).fit(X, Y)
print('score with training data = ',bl_model.score(X,Y))
```

    score with training data =  0.8361391694725028
    


```python
org_test_data = pd.read_csv('test.csv')
test_data = org_test_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

mean_fare = data['Fare'].mean()

values = {'Age' : mean_age, 'Fare': mean_fare, 'Embarked': 'S'}
test_data = test_data.fillna(value=values)
```


```python
test_data['Sex'] = test_data.Sex.apply(lambda x: int(x =='male'))
test_data['Emb_C'] = test_data.Embarked.apply(lambda x: int(x =='C'))
test_data['Emb_Q'] = test_data.Embarked.apply(lambda x: int(x =='Q'))
test_data['Emb_S'] = test_data.Embarked.apply(lambda x: int(x =='S'))

test_data = test_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Emb_C', 'Emb_Q', 'Emb_S']]
```


```python
predictions = best_model.predict(test_data)
output = pd.DataFrame({'PassengerId': org_test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
```

    Your submission was successfully saved!
    

Score for submit this prediction to Kaggle = 0.77751

## Conclusion
For this problem first we look at overview of data we have. Then, we clean it up (select only useful feature fill null data). After that we do analysis about corrlation and outlier. And, we build classification model with random forest tree and tune it to get the best model. Last, we use our best model to do prediction and submit it to Kaggle.

Actually, I also try Adaboost but it worst than this. But I still think that there are better approch to this problem which I havn't try it.

github : https://github.com/Satjarporn/Titanic


```python

```
