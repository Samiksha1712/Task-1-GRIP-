#!/usr/bin/env python
# coding: utf-8

# # SUPERVISED MACHINE LEARNING (LINEAR REGRESSION)

# Author = Samiksha Pundlik Kalamkar

# The Sparks Foundation (Data science and business analytics)(GRIP APR21)

# # Import libraries

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading the data from the link provided in GRIP Task pdf 

# In[26]:


link= "http://bit.ly/w-data"
s_data=pd.read_csv(link)
print("Data imported Succesfully")


# In[28]:


s_data.head(9)


# In[29]:


s_data.tail(9)


# In[30]:


#shape gives the shape of an array


# In[31]:


s_data.shape


# In[32]:


#Describe() function shows count, mean, std, minimum, percentiles & maximum


# In[33]:


s_data.describe()


# In[34]:


#info() function to get information about the data


# In[35]:


s_data.info()


# In[36]:


s_data.isnull().sum()


# In the dataset there are 25 students, study hours and marks are already given.
# we have to predict the percentage of students based on number of hours the student studies.

# # DATA VISUALIZATION

# # To plot box plot

# In[37]:


plt.boxplot(s_data)
plt.show()


# # To plot score distribution

# In[38]:


s_data.plot(x='Hours', y='Scores', style='r.')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')
plt.grid(True)
plt.show()


# # Conclusion= The graph clearly shows positive linear relation between number of hours studied and percentage of score.

# # PREPARING THE DATA

# In[44]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  


# # Splitting the data using train test split

# In[45]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# # TRAINING THE ALGORITHM

# In[46]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# # Plot the test data.

# In[50]:


line = regressor.coef_*X+regressor.intercept_


plt.scatter(X, y)
plt.plot(X, line);
plt.xlabel("Hours",fontsize=15)
plt.ylabel("Scores",fontsize=15)
plt.title("Regression line(Train set)",fontsize=15)
plt.show()


# # Calculating the accuracy of model on train dataset

# In[52]:


plt.scatter(X_train,y_train)
print("Train set Score")
print(regressor.score(X_train,y_train))
plt.xlabel("Hours",fontsize=15)
plt.ylabel("Scores",fontsize=15)
plt.title("Regression line(Test set)",fontsize=15)
plt.show()


# # Calculating the accuracy of model on Test dataset

# In[51]:


print("Test Score")
print(regressor.score(X_test,y_test))
plt.scatter(X_test,y_test)
plt.plot(X_train,regressor.predict(X_train),color="r")
plt.xlabel("Hours",fontsize=15)
plt.ylabel("Scores",fontsize=15)
plt.title("Regression line(Test set)",fontsize=15)
plt.show()


# # Making Predictions

# In[18]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# # Compraing actual vs predicted scores

# In[19]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# # Calculating the predicting score of the model

# In[20]:


hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# # Calculating mean absolute error of the module

# In[21]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('R2 Score:',metrics.r2_score(y_test,y_pred))


# # So, we can conclude that if a student studies 9.25 hours a day his predicted score will  be 93.69% 

# # TO SAVE THE MODULE

# In[22]:


import joblib
joblib.dump(regressor,"TASK1.pckl")


# In[23]:


model=joblib.load("TASK1.pckl")


# In[ ]:




