#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np 
import missingno as msno
import matplotlib.pyplot as plt
import ast
import plotly.express as px
import seaborn as sns
from datetime import date as dt
import calendar
import datetime as dt

import warnings
warnings.filterwarnings('ignore')
import plotly.express as px


# In[3]:


#Read  Customer data input 
cus_data_raw = pd.read_csv("C:\\Users\\Namana\\OneDrive\\Desktop\\Projects\\Customer_Segmentation\\dataset_for_analyst_assignment_20201120.csv") 


# In[4]:


cus_data_raw['PREFERRED_RESTAURANT_TYPES'] = cus_data_raw['PREFERRED_RESTAURANT_TYPES'].str.replace('\n', '')
cus_data_raw['PREFERRED_RESTAURANT_TYPES'] = cus_data_raw['PREFERRED_RESTAURANT_TYPES'].str.replace(' ', '')
cus_data_raw["preferred_American_restaurant"] = cus_data_raw["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "american" in str(row).lower() else False)
cus_data_raw["preferred_Japanese_restaurant"] = cus_data_raw["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "japanese" in str(row).lower() else False)
cus_data_raw["preferred_Italian_restaurant"] = cus_data_raw["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "italian" in str(row).lower() else False)
cus_data_raw["preferred_Mexican_restaurant"] = cus_data_raw["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "mexican" in str(row).lower() else False)
cus_data_raw["preferred_Indian_restaurant"] = cus_data_raw["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "indian" in str(row).lower() else False)
cus_data_raw["preferred_Middleeastern_restaurant"] = cus_data_raw["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "middleeastern" in str(row).lower() else False)
cus_data_raw["preferred_Korean_restaurant"] = cus_data_raw["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "korean" in str(row).lower() else False)
cus_data_raw["preferred_Thai_restaurant"] = cus_data_raw["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "thai" in str(row).lower() else False)
cus_data_raw["preferred_Vietnamese_restaurant"] = cus_data_raw["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "vietnamese" in str(row).lower() else False)
cus_data_raw["preferred_Hawaiian_restaurant"] = cus_data_raw["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "hawaiian" in str(row).lower() else False)
cus_data_raw["preferred_Greek_restaurant"] = cus_data_raw["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "greek" in str(row).lower() else False)
cus_data_raw["preferred_Spanish_restaurant"] = cus_data_raw["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "spanish" in str(row).lower() else False)
cus_data_raw["preferred_Nepalese_restaurant"] = cus_data_raw["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "nepalese" in str(row).lower() else False)
cus_data_raw["preferred_Chinese_restaurant"] = cus_data_raw["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "chinese" in str(row).lower() else False)

#I would like to validate that any preffered resturatnt types  are  not left by changing values 
cus_data_raw['PREFERRED_RESTAURANT_TYPES'] = cus_data_raw['PREFERRED_RESTAURANT_TYPES'].str.replace('american', '0')
cus_data_raw['PREFERRED_RESTAURANT_TYPES'] = cus_data_raw['PREFERRED_RESTAURANT_TYPES'].str.replace('japanese', '1')
cus_data_raw['PREFERRED_RESTAURANT_TYPES'] = cus_data_raw['PREFERRED_RESTAURANT_TYPES'].str.replace('italian', '2')
cus_data_raw['PREFERRED_RESTAURANT_TYPES'] = cus_data_raw['PREFERRED_RESTAURANT_TYPES'].str.replace('mexican', '3')
cus_data_raw['PREFERRED_RESTAURANT_TYPES'] = cus_data_raw['PREFERRED_RESTAURANT_TYPES'].str.replace('indian', '4')
cus_data_raw['PREFERRED_RESTAURANT_TYPES'] = cus_data_raw['PREFERRED_RESTAURANT_TYPES'].str.replace('middleeastern', '5')
cus_data_raw['PREFERRED_RESTAURANT_TYPES'] = cus_data_raw['PREFERRED_RESTAURANT_TYPES'].str.replace('korean', '6')
cus_data_raw['PREFERRED_RESTAURANT_TYPES'] = cus_data_raw['PREFERRED_RESTAURANT_TYPES'].str.replace('thai', '7')
cus_data_raw['PREFERRED_RESTAURANT_TYPES'] = cus_data_raw['PREFERRED_RESTAURANT_TYPES'].str.replace('vietnamese', '8')
cus_data_raw['PREFERRED_RESTAURANT_TYPES'] = cus_data_raw['PREFERRED_RESTAURANT_TYPES'].str.replace('hawaiian', '9')
cus_data_raw['PREFERRED_RESTAURANT_TYPES'] = cus_data_raw['PREFERRED_RESTAURANT_TYPES'].str.replace('greek', '10')
cus_data_raw['PREFERRED_RESTAURANT_TYPES'] = cus_data_raw['PREFERRED_RESTAURANT_TYPES'].str.replace('spanish', '11')
cus_data_raw['PREFERRED_RESTAURANT_TYPES'] = cus_data_raw['PREFERRED_RESTAURANT_TYPES'].str.replace('nepalese', '12')
cus_data_raw['PREFERRED_RESTAURANT_TYPES'] = cus_data_raw['PREFERRED_RESTAURANT_TYPES'].str.replace('chinese', '13')
#cus_data_raw.PREFERRED_RESTAURANT_TYPES.value_counts()

#droping the column 
cus_data_raw.drop('PREFERRED_RESTAURANT_TYPES', axis=1,inplace=True)


# In[5]:



store_columns = []                   # creating an empty data to store new data columns
#cus_data_raw=cus_data_raw.reset_index()
for i in cus_data_raw.PURCHASE_COUNT_BY_STORE_TYPE:
    store_columns.append(eval(i))  # using eval() to transform the type of data into usable form (as a dictionary in this case )

df = pd.DataFrame(store_columns)    # turning it into a new dataframe
cus_data_1 = pd.concat([cus_data_raw,df],axis=1)  # concatenating two dataframes with the same indexes


# In[6]:


# Renaming some columns having spaces between words and making it easir for usage later in the analysis or code writing
cus_data_1.rename(columns ={'General merchandise':'GENERAL_MERCHANDISE','Pet supplies':'PET_SUPPLIES',
                             'Retail store':'RETAIL_STORE','Grocery':'GROCERY','Restaurant':'RESTAURANT'},inplace = True)
cus_data_raw=cus_data_1.drop(columns=['PURCHASE_COUNT_BY_STORE_TYPE'])


# ## Data Understanding

# In[7]:


cus_data_raw.shape


# In[8]:


cus_data_raw .info()

cus_data_raw['REGISTRATION_DATE']=pd.to_datetime(cus_data_raw['REGISTRATION_DATE'])
cus_data_raw['FIRST_PURCHASE_DAY']=pd.to_datetime(cus_data_raw['FIRST_PURCHASE_DAY'])
cus_data_raw['LAST_PURCHASE_DAY']=pd.to_datetime(cus_data_raw['LAST_PURCHASE_DAY'])
# In[9]:


# Creating new "USER_TYPE" column with the following conditions:

cus_data_raw.loc[(cus_data_raw.PURCHASE_COUNT > 1) & (cus_data_raw.AVG_DAYS_BETWEEN_PURCHASES > 30) ,
                 'USER_TYPE' ] = "Churning users"
cus_data_raw.loc[(cus_data_raw.PURCHASE_COUNT > 1) & (cus_data_raw.AVG_DAYS_BETWEEN_PURCHASES <= 30),
                 'USER_TYPE'] = "Active users"
cus_data_raw.loc[cus_data_raw.PURCHASE_COUNT == 1, 'USER_TYPE'] = "First-time or one time users"
cus_data_raw.loc[cus_data_raw.PURCHASE_COUNT == 0, 'USER_TYPE'] = "Inactive users"


# In[10]:


cus_data_raw.duplicated().sum()


# In[11]:


datacheck1=cus_data_raw[cus_data_raw['PURCHASE_COUNT']==0]
len(datacheck1)


# In[12]:


#Missing values of each column 
print(datacheck1.isna().sum())


# <span style='color:blue'> 
# Comments:
#     
#  -User ID, Registration Date, Registation Country, Purchase Count, Preferred devide, User has valid payement, Purchase count
#     by valid type are completely populated. <br>
# -We have very less data for preferred restaurant type. /n
# -Rest of the columns are mostly missing 9955 values.   <br>
# -There are no duplicates  <br>
#  - I do think that there are alot of redundancy with respect to the columns (features) like min and max purchase euros, lunch, dinner, breakfast purchases and alot more. It can be derived out of some other columns as well. I still kept the column as of now just in case it makes anything easier later. But can be avoided to save some space. 
#     
# How do we handle it ?  </span>

# In[13]:


# Identiy the columns with non- missing values 
msno.bar(cus_data_raw,color='#008000',figsize=(10,5), fontsize=8)


# <span style='color:blue'> 
# If you observe the data there are PURCHASE_COUNT =0 and they have only user id and preferred device data is populated mostly.
# All the above columns where 9955 rows missing values belong to these rows(by observation)
# This could mean the user has installed the app and not made any order.
# These customers anyway are a different target. let us consider this segment is already at our hand.
# To entice potential customers to make a purchase, provide incentives like discounts, complimentary gifts, or free delivery.
# on their first purchase can help them save money and try OUR SERVICE, thereby get used to the comfort.
# On account of the above explanation, we an get rid of the users whose purchse count is zero for now,
# and treat them as a seperate segment </span>

# In[14]:


cus_data=cus_data_raw[cus_data_raw['PURCHASE_COUNT']>0]
len(cus_data)   


# In[15]:


# Identiy the columns with non- missing values 
msno.bar(cus_data,color='#008000',figsize=(10,5), fontsize=8)


# In[16]:


#the columns avg days between purchases and median days between purchases are null
#and both the columns are missing on the same rows
cus_data[cus_data['AVG_DAYS_BETWEEN_PURCHASES'].isnull()].equals(cus_data[cus_data['MEDIAN_DAYS_BETWEEN_PURCHASES'].isnull()])


# In[17]:


#It could be null because there may be only one purchase.Check the difference between last purhase and first purchase to confirm
cus_data['total_days']= pd.to_datetime(cus_data['LAST_PURCHASE_DAY'])- pd.to_datetime( cus_data['FIRST_PURCHASE_DAY'])
cus_data['TOTAL_DAYS_BETWEEN_PURCHASES']=cus_data['total_days'].dt.days
cus_data.drop(columns=['total_days'],inplace=True)


# In[18]:


data_check1=cus_data[(cus_data['AVG_DAYS_BETWEEN_PURCHASES'].isnull()) & (cus_data['MEDIAN_DAYS_BETWEEN_PURCHASES'].isnull())]
len(data_check1)


# In[19]:


data_check1['TOTAL_DAYS_BETWEEN_PURCHASES'].value_counts()


# In[20]:


data_check1['PURCHASE_COUNT'].value_counts()


# <span style='color:blue'> 
# This says that most of the null rows in avg days between purchases and median days 
# between purchases are from  rows with purchase count = 1.
# We can populate the AVG_DAYS_BETWEEN_PURCHASES and MEDIAN_DAYS_BETWEEN_PURCHASES 4179 of rows with purchase count 1 with 0 
# We can get rid of the 17 rows with purchase count 2 and having null values in the days between
# purchases or keep them ( <1% of data) as it might not add much value
# Hence, It is safe to induce 0 into the cells where AVG_DAYS_BETWEEN_PURCHASES and MEDIAN_DAYS_BETWEEN_PURCHASES is null  </span>

# In[21]:


cus_data['AVG_DAYS_BETWEEN_PURCHASES'].fillna(0, inplace=True)
cus_data['MEDIAN_DAYS_BETWEEN_PURCHASES'].fillna(0, inplace=True)


# In[22]:


len(cus_data)


# In[23]:


msno.bar(cus_data,color='#008000',figsize=(10,5), fontsize=8)


# In[24]:


len(cus_data)


# ## Exploratory Data Analysis

# In[25]:


cus_data_raw.USER_TYPE.value_counts(normalize=True).mul(100).round(0)


# In[26]:


fig = px.pie(cus_data_raw.USER_TYPE.value_counts(), values='USER_TYPE', 
             names=['Inactive users','Churning users','First-time users','Active users'],
             title="User's device preferences",width=700, height=400)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# <span style='color:blue'> 
#   There are alot of inactive users 45%. We really need to get lure them into a first time experience. There is a lot of potential in this area </span>

# In[27]:


cus_data.describe()     #ignore user id 


# <span style='color:blue'> 
# Average Purchase count is ~6 per person. 
# Mean POV : Average delivery is ~5 which is a large part and a very less Take away deliveries  </span>

# <span style='color:blue'>  PERSPECTIVE - PREFERRED DEVICE </span>

# In[28]:


cus_data.groupby(['PREFERRED_DEVICE']).agg({
    'USER_ID':'count',
    'PURCHASE_COUNT': 'sum',
    'TOTAL_PURCHASES_EUR': 'sum',
    'IOS_PURCHASES': 'sum',
    'WEB_PURCHASES': 'sum',
    'ANDROID_PURCHASES': 'sum'  
})


# In[29]:


cus_data.PREFERRED_DEVICE.value_counts(normalize=1).mul(100).round(0)


# In[30]:


fig = px.pie(cus_data_raw.PREFERRED_DEVICE.value_counts(), values='PREFERRED_DEVICE', names=['ios','android','web'],
             title="User's device preferences",width=700, height=300)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# In[31]:


cus_data_raw.groupby(['PREFERRED_DEVICE','USER_HAS_VALID_PAYMENT_METHOD']).size()


# In[32]:


#PASSIVE USERS BASED ON DEVICE - TO MAKE NECESSARY IMPROVEMENTS 
cus_data_raw.groupby(['PREFERRED_DEVICE','USER_TYPE']).size()


# <span style='color:blue'> 
# INSIGHT: Wolt app orders predominantly stem from three primary devices:
#         ~44% from iOS, ~39% from Android, and ~17% from the website. 
# Given the prevalent use of mobile phones for food ordering due to their convenience, 
# it is advisable for Wolt to customize user acquisition campaigns and banner designs to align with various phone sizes.
# Specially, preference could be given to iphone features, if any.  </span>

# <span style='color:blue'>  PERSPECTIVE - COUNTRY </span>

# In[33]:


cus_data.REGISTRATION_COUNTRY.value_counts(normalize=1).mul(100).round(0)


# In[34]:


countries= cus_data_raw.REGISTRATION_COUNTRY.value_counts().rename_axis('Countries').reset_index(name='Numbers')
fig = plt.figure(figsize =(30, 10))
plt.bar(countries['Countries'], countries['Numbers'], color ='maroon', width = 0.5)
plt.xlabel('Countries'),
plt.ylabel('Numbers')
plt.title('Distribution of USERS ACROSS COUNTRIES')
plt.show()


# In[35]:


cus_data.describe()


# In[36]:


grouped_data=cus_data.groupby(['REGISTRATION_COUNTRY']).agg({
    'GENERAL_MERCHANDISE': 'sum',
    'GROCERY': 'sum',
    'PET_SUPPLIES': 'sum',
    'RESTAURANT': 'sum',
    'RETAIL_STORE': 'sum',
    'PURCHASE_COUNT':'sum',
    'TOTAL_PURCHASES_EUR': 'sum'       # measure for total purchase in EUR
}).sort_values(by="TOTAL_PURCHASES_EUR", ascending=False).reset_index()
grouped_data.head()


# In[37]:


# Plotting using seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

metrics = ['GENERAL_MERCHANDISE','GROCERY','PET_SUPPLIES','RESTAURANT','RETAIL_STORE']

# Set a custom color palette for each metric
color_palette = {'GENERAL_MERCHANDISE':'Accent_r','GROCERY':'Blues_d','PET_SUPPLIES':'PuBu','RESTAURANT':'jet_r','RETAIL_STORE':'viridis_r'}

for metric in metrics:
    sns.barplot(x='REGISTRATION_COUNTRY', y=metric, data=grouped_data, label=metric, palette=color_palette[metric])

plt.title('Comparison of Metrics by Country')
plt.ylabel('Sum of Metrics')
plt.legend()
plt.show()


# In[38]:



grouped_data.plot(x="REGISTRATION_COUNTRY", y=["TOTAL_PURCHASES_EUR","PURCHASE_COUNT"], kind="bar", figsize=(8, 8),width=1)
 
# print bar graph
plt.show()


# In[39]:


#COUNTRY WISE PASSIVE USERS TO TAKE NECCESSARY ACTION 
cus_data_top3cntry=cus_data_raw[cus_data_raw['REGISTRATION_COUNTRY'].isin(['FIN','DNK','GRC'])]
grouped =cus_data_top3cntry.groupby(['REGISTRATION_COUNTRY','USER_TYPE']).size()


# In[40]:


percentage = round(grouped / len(cus_data_top3cntry) * 100)
percentage


# In[41]:


cus_data_top3cntry.groupby(['REGISTRATION_COUNTRY']).agg({
    'DISTINCT_PURCHASE_VENUE_COUNT': 'mean',
    'BREAKFAST_PURCHASES': 'sum',
    'LUNCH_PURCHASES': 'sum',
    'EVENING_PURCHASES': 'sum',
    'DINNER_PURCHASES': 'sum',
    'LATE_NIGHT_PURCHASES': 'sum' }).reset_index()


# <span style='color:blue'> INSIGHT: Most of the business 99% are highly located in Finland, Denmark, Greece <br>
#  Restaurant segment gets the highest business across all the countries <br>
# followed by Retail Stores and then Groceries <br> 
# Interestingly, though Finland leads in no. of orders, Denmark leads in terms of purchase values i.e.in terms of Euros(Monetary)<br> 
# Denmark users mostly order Dinner where as Finland and Greece users order Lunch slightly more than dinner
# </span>
# 

# <span style='color:blue'>  MOST_COMMON_WEEKDAY_TO_PURCHASE PATTERN </span>

# In[42]:


cus_data.MOST_COMMON_WEEKDAY_TO_PURCHASE.value_counts(normalize=1).mul(100).round(0)
# Group by two columns and calculate the size
grouped = cus_data_top3cntry.groupby(['REGISTRATION_COUNTRY', 'MOST_COMMON_WEEKDAY_TO_PURCHASE']).size().reset_index(name='Count')


# In[43]:


# Create a pivot table for heatmap
heatmap_data = grouped.pivot('REGISTRATION_COUNTRY', 'MOST_COMMON_WEEKDAY_TO_PURCHASE', 'Count')

# Plotting the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='g', cbar_kws={'label': 'Count'})
plt.title('Most Common Weekday to Purchase by Registration Country')
plt.show()


# <span style='color:blue'>  There is no significant demarcation here. 
#     Tuesday is comparatively less preferred by people for ordering
# </span>

# <span style='color:blue'>  MOST_COMMON_HOUR_OF_THE_DAY_TO_PURCHASE PATTERN </span>
# 

# In[44]:


# Creating new "USER_TYPE" column with the following conditions:
cus_data.loc[(cus_data.MOST_COMMON_HOUR_OF_THE_DAY_TO_PURCHASE >= 0) 
             & (cus_data.MOST_COMMON_HOUR_OF_THE_DAY_TO_PURCHASE < 5), 'time'] = "Mid_Night"
cus_data.loc[(cus_data.MOST_COMMON_HOUR_OF_THE_DAY_TO_PURCHASE >= 5) 
             & (cus_data.MOST_COMMON_HOUR_OF_THE_DAY_TO_PURCHASE < 11), 'time'] = "Morning"
cus_data.loc[(cus_data.MOST_COMMON_HOUR_OF_THE_DAY_TO_PURCHASE >= 11)
             & (cus_data.MOST_COMMON_HOUR_OF_THE_DAY_TO_PURCHASE < 16), 'time'] = "Lunch"
cus_data.loc[(cus_data.MOST_COMMON_HOUR_OF_THE_DAY_TO_PURCHASE >= 16) 
             & (cus_data.MOST_COMMON_HOUR_OF_THE_DAY_TO_PURCHASE <= 23), 'time'] = "Night"


# In[45]:


cus_data.time.value_counts(normalize=1).mul(100).round(0)


# In[46]:


a=  cus_data_top3cntry.groupby(['REGISTRATION_COUNTRY']).agg({
'BREAKFAST_PURCHASES': 'sum',
'LUNCH_PURCHASES': 'sum',
'EVENING_PURCHASES': 'sum',
'DINNER_PURCHASES': 'sum',
'LATE_NIGHT_PURCHASES': 'sum'   }).reset_index()
a.head()


# In[47]:



a.plot(x="REGISTRATION_COUNTRY", y=["BREAKFAST_PURCHASES","LUNCH_PURCHASES","EVENING_PURCHASES","DINNER_PURCHASES","LATE_NIGHT_PURCHASES"], kind="bar", figsize=(8, 6))
 
# print bar graph
plt.show()


# <span style='color:blue'> INSIGHT: Most of the orders come at night between 4 PM to 11PM <br>
#    Denmark users mostly order Dinner where as Finland and Greece users order Lunch slightly more than dinner  </span>
# 
# 

# <span style='color:blue'>  PERSPECTIVE - USER AND PURCHASE COUNT </span>

# In[48]:


cus_data_raw.PURCHASE_COUNT.value_counts()


# In[49]:


cus_data_raw.PURCHASE_COUNT.value_counts(normalize=1).mul(100).round(0)


# In[50]:


cus_data.head()


# In[51]:


cus_data.groupby(['USER_TYPE']).agg({
    'USER_ID': 'count',
    'PURCHASE_COUNT': 'sum',
    'TOTAL_PURCHASES_EUR': 'sum',
    'IOS_PURCHASES' :'sum',
    'WEB_PURCHASES': 'sum',
    'ANDROID_PURCHASES':'sum',
    'PURCHASE_COUNT_DELIVERY':'sum',
    'PURCHASE_COUNT_TAKEAWAY':'sum'
})


# In[52]:


### We just calculate based on the users who ordered the services at least one time
total_sales = cus_data['TOTAL_PURCHASES_EUR'].sum()
total_orders = cus_data['PURCHASE_COUNT'].sum()
AOV = total_sales/total_orders
PF = total_orders/len(cus_data[cus_data.PURCHASE_COUNT >0])
print(f"Avg Order Value: {AOV:0.1f} euros and Purchase Frequency: {PF:0.1f} purchase times" )


# <span style='color:blue'> 
# 
# Comments: There are 04 groups of customers at glance to be analysed & assessed:
# 
# 1. Non-users. [9955 - 45%] - Users who installed the app and never made orders 
# 2. Once-users. [4179 - 19%] - Users whose purchase count was one and never came back
# 3. Repeated-users (Pareto principle: 80% sales figures coming from 20% of this customer group) [7849 - 37%]  <br>
# Delivery is highly likely than take away in all kind of user type
# 
#    
# </span>

# <span style='color:blue'>  IMMEDIATE USERS </SPAN>

# In[53]:


### 
registers = cus_data.groupby(['REGISTRATION_DATE'])['USER_ID'].count().reset_index()
registers.rename(columns = {'USER_ID' : 'registers'}, inplace = True)

## Calculate the users who have registered & immediately used the service
users = cus_data[(cus_data.PURCHASE_COUNT > 0) & (cus_data['REGISTRATION_DATE'] 
                                                  == cus_data['FIRST_PURCHASE_DAY'])].groupby(['REGISTRATION_DATE'])['USER_ID'].count().reset_index()
users.rename(columns = {'USER_ID' : 'immediate_users'}, inplace = True)

group_registers = pd.merge(registers, users, on='REGISTRATION_DATE', how='left')
group_registers['%immediate_users'] = (group_registers['immediate_users'] / group_registers['registers'])*100
###
#group_registers = str_to_date(group_registers, col = 'REGISTRATION_DATE' )
group_registers.head()


# In[54]:


group_registers['%immediate_users'].mean()

print(f"Avg percentage of registers immediately using the Wolt services: {group_registers['%immediate_users'].mean():0.2f}%. ")


# <span style='color:blue'> RESTAURANT TYPE </SPAN>

# In[55]:


def generate_value_counts_for_restaurant_style_preferance(df,column_list):
    pieces=[]
    for col in column_list:
        tmp_series = df[col].value_counts()
        tmp_series.name = col
        pieces.append(tmp_series)
    df_value_counts = pd.concat(pieces, axis=1)
    df_value_counts = df_value_counts .rename_axis('yes_I_prefer')
    return df_value_counts


# In[56]:


preferred_restaurants=["preferred_American_restaurant", "preferred_Italian_restaurant", "preferred_Japanese_restaurant","preferred_Mexican_restaurant",
                                "preferred_Indian_restaurant","preferred_Middleeastern_restaurant","preferred_Korean_restaurant","preferred_Thai_restaurant"
                                 ,"preferred_Vietnamese_restaurant","preferred_Hawaiian_restaurant","preferred_Greek_restaurant","preferred_Spanish_restaurant",
                                 "preferred_Nepalese_restaurant","preferred_Chinese_restaurant"]

value_counts_res = generate_value_counts_for_restaurant_style_preferance(cus_data_raw,preferred_restaurants)
#value_counts_res = value_counts_res.rename_axis('yes_I_prefer')

value_counts_res 


# In[57]:


#general 
#restaurant preference 
#User type angle 
#country type angle 
fig = px.bar(value_counts_res.iloc[1:, :],  y=["preferred_American_restaurant", "preferred_Italian_restaurant", "preferred_Japanese_restaurant","preferred_Mexican_restaurant",
                                "preferred_Indian_restaurant","preferred_Middleeastern_restaurant","preferred_Korean_restaurant","preferred_Thai_restaurant"
                                 ,"preferred_Vietnamese_restaurant","preferred_Hawaiian_restaurant","preferred_Greek_restaurant","preferred_Spanish_restaurant",
                                 "preferred_Nepalese_restaurant","preferred_Chinese_restaurant"] ,title="Restaurant Style Preferences of Users",text_auto=True)
fig.update_layout(barmode='group')
fig.show()


# <span style='color:blue'> People mostly prefer American Restuarants followed by Italian and then Japanese. <br>
# Spanish and Nepalese Restaurants are very less preferred by people </span>

# ## RFM Analysis

# <span style='color:blue'> Customer Value RFM Model: We aim to initiate a straightforward analysis. <br>
#     To achieve this, we will employ the RFM (Recency, Frequency, and Monetary Value) model for customer segmentation. <br>
#     The RFM model involves assessing each customer's transactions to derive three key attributes: <br>
# Recency: Indicates the time elapsed since a customer's most recent purchase. <br>
# Frequency: Measures the regularity of a customer's transactions.  <br>
# Monetary Value: Quantifies the total monetary worth of all customer transactions. <br>
# This approach allows us to distill complex customer data into actionable insights,
# providing a foundation for targeted strategies based on customer behavior. </span>

# In[58]:


def return_date(df, col):
    
    df[col]= pd.to_datetime(df[col])
    df[col] = df[col].apply(lambda x: x.date()) # return a column for certain date in year - 2015-12-10
    
    return df
customers = return_date(cus_data, col = 'LAST_PURCHASE_DAY')


# In[59]:


cus_data['days_last_PO'] = dt.date(2020,10,31) - cus_data['LAST_PURCHASE_DAY'] 
customers['days_last_PO'] = customers['days_last_PO'].apply(lambda x: x.days)
len(cus_data)


# In[60]:


cus_RFM =cus_data[['USER_ID','days_last_PO', 'PURCHASE_COUNT', 'TOTAL_PURCHASES_EUR'
                   , 'AVG_PURCHASE_VALUE_EUR']] 
cus_RFM.columns = ['user_id','recency', 'frequency',  'TOTAL_PURCHASES_EUR','monetary']


# In[61]:


cus_RFM.head()


# In[62]:


for col in cus_RFM.columns:
    print(col)


# In[63]:


cus_RFM.info()


# In[64]:


#top cstomers with respect to frequency  and see if they have gone out 
quantiles = cus_RFM.quantile(q=[0.25,0.5,0.75])
quantiles= quantiles.to_dict()

# Function arguments (x = value, p = RECENCY, MONETARY, FREQUENCY, d = quartiles dict) to create Wolt RFM segments in RECENCY
def RScoring(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1
# Function arguments (x = value, p = RECENCY, MONETARY, FREQUENCY, d = quartiles dict) 
#to create Wolt RFM segments in FREQUENCY AND MONETARY
def FMScoring(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
# Creating a Wolt_RFMScores segmentation table so we can evaluate the analysis
Wolt_RFMSegment = cus_RFM.copy()
Wolt_RFMSegment['R_Score'] = Wolt_RFMSegment['recency'].apply(RScoring, args=('recency',quantiles,))
Wolt_RFMSegment['F_Score'] = Wolt_RFMSegment['frequency'].apply(FMScoring, args=('frequency',quantiles,))
Wolt_RFMSegment['M_Score'] = Wolt_RFMSegment['monetary'].apply(FMScoring, args=('monetary',quantiles,))

Wolt_RFMSegment.head()


# In[65]:


Wolt_RFMSegment[Wolt_RFMSegment['R_Score']==1].sort_values(by='TOTAL_PURCHASES_EUR',ascending=False).head(10)


# As a non Data Scientist, I could evaluate different quantiles for recency, frequency and monetary values and bucket them into differnt groups manually like High value but churned customers, Low value Less recent Customers and so on 
# As a Data Scientist I would use Machine Learning Algorithm - K Means Clustering since it is much more scalable and Interpretable

# ### Univariate Analysis 

# In[66]:


x=cus_RFM['recency']
ax=sns.distplot(x)


# In[67]:


fig = plt.figure(figsize =(30, 10))
x=cus_RFM['frequency']
ax=sns.distplot(x)


# In[68]:


fig = plt.figure(figsize =(30, 10))
x=cus_RFM['monetary']
ax=sns.distplot(x)


# ### Bivariate Analysis 

# In[69]:


sns.scatterplot(data=cus_RFM, x='recency',y='frequency' )


# In[70]:


cus_RFM[cus_RFM['frequency'] > 300]       #We might have lost this Golden Customer


# In[71]:


sns.scatterplot(data=cus_RFM, x='recency',y='monetary' )


# In[72]:


sns.scatterplot(data=cus_RFM, x='frequency',y='monetary' )


# In[73]:


sns.heatmap( cus_RFM.corr(),annot=True,cmap='coolwarm')

Frequency and Total Purchase Euros are highly corelated. 
Hence to avoid reduendancy we shall drop Total purchase Euros and rather consider Avg purchase value as a monetary parameter
# # K Means Clustering

# In[74]:


from sklearn.preprocessing import StandardScaler
# create new dataframe with transformed values
df_t =  cus_RFM.copy()

ss = StandardScaler()
df_t['recency_t'] = ss.fit_transform( cus_RFM['recency'].values.reshape(-1,1))
df_t['frequency_t'] = ss.fit_transform( cus_RFM['frequency'].values.reshape(-1,1))
df_t['monetary_t'] = ss.fit_transform( cus_RFM['monetary'].values.reshape(-1,1))
#df_t['avg_monetary_t'] = ss.fit_transform( cus_RFM['TOTAL_PURCHASES_EUR'].values.reshape(-1,1))


# In[75]:


df_t.info(10)


# In[76]:


df_t.to_csv("C:\\Users\\Namana\\OneDrive\\Desktop\\Projects\\Customer_Segmentation\\df_t.csv")


# In[77]:


# Get rid of the row where recency is null 
df_t=df_t[df_t['recency'].notnull()].reset_index()


# In[78]:


len(df_t)


# In[79]:


from sklearn.cluster import KMeans


# ###  find the K value using Elbow method

# In[80]:


intertia_scores=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_t[['recency_t','frequency_t','monetary_t']])
    intertia_scores.append(kmeans.inertia_)
plt.plot(range(1,11),intertia_scores,'--bo')


# In[81]:


#  Looks like K=4 should be our choice 


# In[82]:


clustering2 = KMeans(n_clusters=4)
identified_clusters=clustering2.fit(df_t[['recency_t','frequency_t','monetary_t']])
df_t['Kmeans_segmentation'] =clustering2.labels_
df_t.head()


# In[83]:


centers =pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['recency_t','frequency_t','monetary_t']


# In[84]:


df_t.groupby(['Kmeans_segmentation']).agg({
'recency': 'mean',
'frequency': 'mean',
'monetary': 'mean',
'user_id': 'count'}).sort_values(by="frequency", ascending=False)


# In[85]:


cus_segment=cus_data_raw.merge(df_t.rename({'user_id': 'user_id_r'}, axis=1),
               left_on='USER_ID', right_on='user_id_r', how='left')
cus_segment['Kmeans_segmentation'].fillna('4',inplace = True)


# In[86]:


cus_segment['Segment']=np.where(cus_segment['Kmeans_segmentation']==2,'Loyal Customer',
                                np.where(cus_segment['Kmeans_segmentation']==0,'Potential Loyal Customer',
                                        np.where(cus_segment['Kmeans_segmentation']==3,'High value Churning Customers',
                                              np.where(cus_segment['Kmeans_segmentation']==1,'Lost Customer','Never Ordered'))))


# In[87]:


cus_segment.to_csv("C:\\Users\\Namana\\OneDrive\\Desktop\\Projects\\Customer_Segmentation\\segment_output_final.csv")


# In[88]:


fig = px.scatter_3d(df_t, x='recency', y='frequency', z='monetary',color='Kmeans_segmentation',
                    height=700,width=700,opacity=0.5,size_max=0.1,template='plotly_white',
                  )
fig.update_layout(title_text='Customers across RFM Clusters', title_x=0.5)
fig.show()