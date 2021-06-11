#!/usr/bin/env python
# coding: utf-8

# You work for the online store Ice, which sells video games all over the world. User and expert reviews, genres, platforms (e.g. Xbox or PlayStation), and historical data on game sales are available from open sources. You need to identify patterns that determine whether a game succeeds or not. This will allow you to spot potential big winners and plan advertising campaigns.

# **Step 1. Open the data file and study the general information**
# 
# **Step 2. Prepare the data**
# 
# 
#     •Replace the column names (make them lowercase).
#     
#     •Convert the data to the required types.
#     
#     •Describe the columns where the data types have been changed and why.
#     
#     •deal with missing values:
#     
#   
#     
# **Step 3. Analyze the data**
# 
#     •Look at how many games were released in different years. Is the data for every period significant?
#     
#     •Look at how sales varied from platform to platform. Choose the platforms with the greatest total sales and build a distribution based on data for each year. Find platforms that used to be popular but now have zero sales. How long does it generally take for new platforms to appear and old ones to fade?
#     
#     •Determine what period you should take data for. To do so, look at your answers to the previous questions. The data should allow you to build a prognosis for 2017.
#     
#     •Work only with the data that you've decided is relevant. Disregard the data for previous years.
#     
#     •Which platforms are leading in sales? Which ones are growing or shrinking? Select several potentially profitable platforms.
#     
#     •Build a box plot for the global sales of all games, broken down by platform. Are the differences in sales significant? What about average sales on various platforms? Describe your findings.
#     
#     •Take a look at how user and professional reviews affect sales for one popular platform (you choose). Build a scatter plot and calculate the correlation between reviews and sales. Draw conclusions.
#     
#     •Keeping your conclusions in mind, compare the sales of the same games on other platforms.
#     
#     •Take a look at the general distribution of games by genre. What can we say about the most profitable genres? Can you generalize about genres with high and low sales?
# 
# **Step 4. Create a user profile for each region**
# 
#     ***For each region (NA, EU, JP), determine:***
# 
#     •The top five platforms. Describe variations in their market shares from region to region.
#     
#     •The top five genres. Explain the difference.
#     
#     •Do ESRB ratings affect sales in individual regions?
#     
# **Step 5. Test the following hypotheses:**
# 
#     •Average user ratings of the Xbox One and PC platforms are the same.
#     
#     •Average user ratings for the Action and Sports genres are different.
#     
# **Overall Conclusion**
#     
# 

# # Open the data file and study the general information

# In[1]:


get_ipython().system(' pip install missingno')


# In[2]:


import pandas as pd
from matplotlib import pyplot as plt 
import math
import numpy as np
from scipy import stats as st
import seaborn as sns
import sys
import warnings
if not sys.warnoptions:
       warnings.simplefilter("ignore")
import missingno as msno


# In[3]:


df_games= pd.read_csv('/datasets/games.csv')


# In[4]:


df_games.info()


# In[5]:


msno.matrix(df_games);
plt.title('Missing values in given data', fontdict={'size':20});
plt.ylabel('Order', fontdict={'size':20});
plt.xlabel('Columns', fontdict={'size':20});


# >There are very high number of missing values in 'critic_score', 'user_suser' and 'Rating' columns. Likewise there are some missing vales in 'Year_of_release'.

# In[6]:


df_games.head()


# In[7]:


for i in df_games.columns:
    print(i, len(df_games[df_games[i]==0]))


# > There are zero values in sales which is natural.

# In[8]:


df_games['Platform'].unique()


# In[9]:


df_games.describe(include=['object'])


# > PS2 is most popular platform and there are 2424 rating which has to be determined

# In[10]:


df_games.User_Score.value_counts()


# In[11]:


df_games.nlargest(5, ['NA_sales'])


# In[12]:


print(df_games.User_Score.value_counts()/len(df_games))


# In[13]:


print(df_games[df_games['User_Score']== 'tbd']['Year_of_Release'].value_counts())


# >Most values of rating to be determined are on 2009

# In[14]:


df_games.Rating.unique()


# In[15]:


df_games.isnull().sum()


# > Most missing values are in 'critic_score', 'user_score' and 'rating' columns

# **Conclusion**

# > Data is downloaded and observed its size, missing values, rows

# # Step 2. Prepare the data

# ***Replace the column names (make them lowercase).***

# In[16]:


df_games.columns=df_games.columns.str.lower()


# In[17]:


df_games[df_games.isnull().any(axis=1)].sample(10)


# **Convert the data to the required types.**

# In[18]:


#df_games['year_of_release'] = pd.to_numeric(df_games['year_of_release'], downcast ='float')
df_games['na_sales'] = pd.to_numeric(df_games['na_sales'], downcast='float')
df_games['eu_sales'] = pd.to_numeric(df_games['eu_sales'], downcast='float')
df_games['jp_sales'] = pd.to_numeric(df_games['jp_sales'], downcast='float')
df_games['other_sales'] = pd.to_numeric(df_games['other_sales'], downcast='float')
df_games['year_of_release'] = pd.to_numeric(df_games['year_of_release'], downcast ='float')
df_games['critic_score'] = pd.to_numeric(df_games['critic_score'], downcast='float')


# In[19]:


df_games.info()


# > Numeric columns are converted from float64 type to float32 type. This step sace significant amount of memory.

# In[20]:


df_games= df_games.dropna(subset= ['name', 'genre'])
df_games


# > Rows with missing name and missing genre are dropped as missing values were only two on these columns column.

# In[21]:


df_games['year_of_release'] = df_games['year_of_release'].fillna(df_games.groupby('name')['year_of_release'].transform('median'))


# In[22]:


df_games['year_of_release'] = df_games['year_of_release'].fillna(df_games.groupby(['genre','platform'])['year_of_release'].transform('median'))


# In[23]:


df_games['year_of_release']=df_games['year_of_release'].astype(int)


# In[24]:


df_games.info()


# > year_of_released changed to int64 type and has no missing values

# In[25]:


rating_grouped=df_games.groupby('genre')['rating'].agg(pd.Series.mode).reset_index()
rating_dict= dict(zip(rating_grouped.genre, rating_grouped.rating))
rating_dict


# In[26]:


df_games['rating']=df_games['rating'].fillna(df_games['genre'].map(rating_dict))


# In[27]:


df_games.info()


# > missing value of rating are filled

# In[28]:


df_games.sample(15)


# In[29]:


len(df_games.name.unique())


# In[30]:


len(df_games[df_games['user_score'].isnull()])


# In[31]:


df1=df_games[df_games['user_score'] != 'tbd']


# In[32]:


#user_score=df1.groupby('genre')['user_score'].agg(pd.Series.mode).reset_index()


# In[33]:


#list_user_score={}
#user_score_dict= dict(zip(user_score.genre, user_score.user_score))
#for key in user_score_dict:
 #   user_score_dict[key]=user_score_dict[key][-1]
#user_score_dict


# In[34]:


# mode of user_score are determined for different genre and assigned to the dictionary


# In[35]:


#df_games['critic_score'] = df_games['critic_score'].fillna(df_games.groupby('genre')['critic_score'].transform('median'))


# In[36]:


#critic_score=df1.groupby('genre')['critic_score'].agg(pd.Series.median).reset_index()
#critic_score_dict= dict(zip(critic_score.genre, critic_score.critic_score))
#for key in critic_score_dict:
#    critic_score_dict[key]=critic_score_dict[key][0]
#critic_score_dict


# In[37]:


df_games['user_score'] = df_games['user_score'].replace({'tbd': np.NaN})


# > Rating to be determined are changed to 'none'

# In[38]:


#df_games['user_score'] = df['user_score'].fillna(df.groupby('genre')['year_of_release'].transform('median'))
#print(df)


# In[39]:


#df_games['user_score']=df_games['user_score'].fillna(df_games['genre'].map(user_score_dict))


# In[40]:


#> Mode of rating in each genre are filled in corresponding missing values of Rating


# In[41]:


df_games['user_score']=df_games['user_score'].astype(float)


# In[42]:


#df_games['critic_score']=df_games['critic_score'].fillna(df_games['genre'].map(critic_score_dict))


# In[43]:


df_games['critic_score']=df_games['critic_score'].astype(float)


# In[44]:


#df_games['critic_score'] = df_games['critic_score'].fillna(df_games.groupby(['genre'])['critic_score'].transform('median'))


# In[45]:


df_games['total_sales']=df_games[['na_sales','eu_sales','jp_sales','other_sales']].sum(axis=1)


# > User Score and Critic score are converted to float type and missing values are filled by corresponding genres median value

# In[46]:


#>total sales (the sum of sales in all regions) for each game is determined and kept these values in a separate column


# **Conclusion**

# > Some missing values are deleted as they were only two. Missing value in _year_of_release_ are filled with median value of
# corresponding genre and converted to int type.
# 
# 
# 
# >Total sales is determined and placed in different column

# In[47]:


#> _Critic_score and user_score_ are filled with mode of corresponding genra.Before that they were converted to float type and 'tbd'
#    was also replace with 'Nan'


# # Analyze Data

# **Look at how many games were released in different years. Is the data for every period significant?**

# In[48]:


df=df_games[['year_of_release','name']].groupby(['year_of_release']).count().sort_values(by='year_of_release').plot(figsize=(8,6),kind='bar');
plt.title('yearwise numbers of games released', fontdict={'size':20});
plt.ylabel('Number of games released');
plt.xlabel('year');


# >> Yearwise number of games . The number of games after year 2000 can be considered as significant

# **•	Look at how sales varied from platform to platform. Choose the platforms with the greatest total sales and build a 
# distribution based on data for each year. Find platforms that used to be popular but now have zero sales. How long does 
# it generally take for new platforms to appear and old ones to fade?**
# 

# In[49]:


df_games['total_sales']=df_games[['na_sales','eu_sales','jp_sales','other_sales']].sum(axis=1)
df_games.sample(7)


# In[50]:


print(df_games.pivot_table(index=['platform',],values='total_sales',aggfunc='sum').sort_values(by='total_sales', ascending=False))


# In[51]:


ax= df_games.pivot_table(index=['platform',],values='total_sales',aggfunc='sum').sort_values(by='total_sales', ascending=False).plot(kind='bar', figsize=(10,8), color='#44ffdd');
plt.title('Sales on each platform', fontdict={'size':20});
plt.ylabel('Total Sales (million)');
plt.xlabel('platform');
df1= df_games.pivot_table(index=['platform',],values='total_sales',aggfunc='sum').sort_values(by='total_sales', ascending=False)
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+.04, i.get_height()+40,             str(round((i.get_height()), 2)), fontsize=11, color='#010fae', rotation=45)
#for row in df1:
 #   ax.text(row.index, row.total_sales+35, s= round(row.total_sales,2), fontdict={'size':10});


# > _PS2_ platform has highest total sales 1255.77 millions.

# In[52]:


df1=df_games[df_games['platform']=='PS2']
print(df1.describe())
len(df1.year_of_release.unique())


# In[53]:


#df1.plot(kind='scatter', x='year_of_release', y='total_sales')
fig,ax=plt.subplots(figsize=(12,8));
ax.vlines(x=df1.year_of_release, ymin=0, ymax=df1.total_sales, color='blue', alpha=0.8, linewidth=10);
#plt.scatter(x=df1.year_of_release,  y=df1.total_sales, s=100, color='red');
ax.set_title(' chart of PS2 platform for different years', fontdict={'size':20});
ax.set_ylabel('total sales (milions)');
ax.set_ylabel('year');
ax.set_xticks(df1.year_of_release);
ax.set_xticklabels(df1.year_of_release, rotation=90, fontdict={'horizontalalignment':'right','size':15});
#for row in df1.itertuples():
#    ax.text(row.index, row.total_sales+5, s= round(row.total_sales,2), fontdict={'size':10});


# > Yearwise sales distribution of all time popular platform

# ****I tried the below code but could not print the text indicating value of vertical line****
# >for row in df1.itertuples():
#     
# >>ax.text(row.index, row.total_sales+5, s= round(row.total_sales,2), fontdict={'size':10});

# In[54]:


df_new= df_games[df_games['year_of_release']>2000]
df2=pd.pivot_table(df_new, index='year_of_release', columns= 'platform', values='total_sales', aggfunc='sum',fill_value=0)
df2.tail(20)


# In[55]:


platforms=df_new[df_new['platform'].isin(['PS2' , 'Wii' , 'PS3' , 'DS' , 'X360' , 'PS4'])]
sns.lineplot(data=platforms, x='year_of_release', y='total_sales', hue='platform', style='platform');
#sns.lineplot(data=fmri, x="timepoint", y="signal", hue="event")
plt.title('Distribution of total sales in different platform', fontdict={'size':20});


# > Platform 'Wii' had very high total sales. Platform PS4 arises in 2013.'DS' platform had highest total sales among all in 2004 and 2005.

# In[56]:


dynamics=df2.T- df2.T.shift(+1, axis=1)
dynamics.sample(3)


# In[57]:


plt.figure(figsize= (17,8))
sns.heatmap(dynamics, cmap='RdBu_r');
plt.title('Heatmap showing changes in  toal_sales in different platforms', fontdict={'size':20});


# >In 2002 only ps2 was doing significantly better than perevious yeara and for  next two year there was no platform doing better than previous year.
# Also, Ps2 also fade out. But in 2004 few platform show slight improvement and DS arises in 2006 doing about increase of sales of about 100 million.
# But unfortunately, it fade out next year. But, Wii rises from nowhere. So, there are cases where someplatform rises suddnely and fade out next year.
# Likewise other are either slowly rising or slowly fading out.
# 
# > Toward the end of the period PS3 has reduced the fall of sales from previous year, PS4's sales is shrinking, Xones's sales is shrinking.
# 
# > Other platform do not have significant change in sales at the end of the period

# •**Determine what period you should take data for. To do so, look at your answers to the previous questions. The data should allow you to build a prognosis for 2017.**

# In[58]:


df_games.info()


# In[59]:


missing_data=df_games[df_games['critic_score'].isnull()]
missing_data.head()
print(missing_data.year_of_release.value_counts().sort_values(ascending=False).head(10))


# > In bar diagram for yearwise number of games released, the number of games released increased continiously up to 2008 and then the number continiously decreased over the given time period. Likewise, in the heatmap most of the platforms sales has decreased from 2008. So, period of 2001 to 2016 should be taken.

# **•	Work only with the data that you've decided is relevant. Disregard the data for previous years**

# **•	Which platforms are leading in sales? Which ones are growing or shrinking? Select several potentially profitable platforms.**

# In[60]:


grouped=df_new.groupby(['platform','year_of_release'])['total_sales'].sum().reset_index()
grouped.sample(7)


# In[61]:


ordered=grouped.groupby(['platform'])['total_sales'].sum().sort_values(ascending=False).reset_index()['platform']
print(grouped.groupby(['platform'])['total_sales'].sum().sort_values(ascending=False).head(7))


# >These are the platform leading in sales. Growing and shrinking platforms are explained in heatmap diagram

# ***•	Build a box plot for the global sales of all games, broken down by platform. Are the differences in sales significant? What about average sales on various platforms? Describe your findings.***

# In[62]:


plt.figure(figsize= (17,8))
sns.boxplot(x='platform', y='total_sales', data=grouped, order=ordered);
plt.title('box plot for the global sales of all games', fontdict={'size':20});


# > In PS2 and DS has very high sales in some year and very low in other. Sales in these platform should have significant variation
# over the period as box height varies very much. Average total sales in platform *PS4, DS, PS3, X360 and PS2* is similar. Also, varion over the 
# year is less in *PS4* among the higher selling platform.

# In[63]:


<div class="alert alert-danger">
<s><b>Reviewer's comment:</b>if not sys.warnoptions:
       warnings.simplefilter("ignore").
</div>


# **•	Take a look at how user and professional reviews affect sales for one popular platform (you choose). Build a scatter plot and calculate the correlation between reviews and sales. Draw conclusions.**

# In[ ]:


df_new.head(5)


# In[ ]:


#hw.plot(x='height', y='weight', kind='scatter') 
#df_new[df_new['user_score'] != 'tbd'].plot(x='user_score',y='total_sales',kind='scatter', alpha=0.7,figsize=(8, 6), grid=True);


# In[ ]:


df_new['critic_score']=df_new['critic_score'].astype(float)


# In[ ]:


print(df_new[df_new.user_score =='tbd'])


# In[ ]:


df_new_user=df_new[df_new['user_score'] !='tbd']
df_new_user['user_score']=df_new_user['user_score'].astype(float)
platform=df_new_user[df_new_user['platform']=='PS2']


# In[ ]:



user_rating=platform[['user_score','total_sales']]
Q11 = platform['total_sales'].quantile(0.25)
Q33 = platform['total_sales'].quantile(0.75)
IQRR = Q33 - Q11
limit1=Q33+ 1.5*IQRR
user_rating=user_rating[user_rating['total_sales']<=limit1]


# In[ ]:


user_rating.info()


# In[ ]:


user_rating.plot(x='user_score',y='total_sales',kind='scatter',alpha=0.6, figsize=(8, 6), grid=True);
plt.title('yearwise numbers of games released', fontdict={'size':20});


# > No significant relation between user _score and total_sales. More points for high 'total sales' when there is high 'user_score' inly indicates user_score is normally high for the games.

# In[ ]:


print(platform.corr(method='spearman')) 


# > co-relation of 0.134735 follow the explanation on scatter diagram

# In[ ]:


platform1= df_new[df_new_user['platform']=='PS2']
critic=platform1[['critic_score','total_sales']]
Q1 = platform1['total_sales'].quantile(0.25)
Q3 = platform1['total_sales'].quantile(0.75)
IQR = Q3 - Q1
limit=Q3+ 1.5*IQR
critic=critic[critic['total_sales']<=limit]


# In[ ]:


critic.plot(x='critic_score',y='total_sales',kind='scatter',alpha=0.4, figsize=(8, 6), grid=True);


# In[ ]:


print(critic['critic_score'].corr(critic['total_sales'], method='spearman'))


# > Small positive corellation between  sales and critic_score. Critic_score seems to have some effect on total sales but do not determine in large extent.

# **•	Keeping your conclusions in mind, compare the sales of the same games on other platforms.**

# In[ ]:


df_new['total_sales']=df_new['total_sales'].sort_values(ascending= False)


# In[ ]:


#other_platform= df_new[df_new_user['platform']== 'PS2' or 'Wii' or 'PS3' or 'DS' or 'X360' or 'PS4']
platforms_new=df_new[df_new['platform'].isin(['PS2' , 'Wii' , 'PS3' , 'DS' , 'X360' , 'PS4','GC'])]
#, 'N64', 'PC', 'PS' ,'PS2', 'PS3', 'PS4', 'PSP','PSV','WS'])]
platforms_new.head(7)
#.sort_values(by='total_sales')

        


# In[ ]:



sales_comparision=pd.pivot_table(platforms_new, index=['name'], columns='platform', values='total_sales' )
sales_comparision=sales_comparision.fillna(0)
sales_comparision.sample(20)


# > Not all the games sales in all these popular platform. The game which sales on two or more platform could have different sales on 
# different platform .
# The difference of sales on different platforms is nominal in some cases ( see: Fight Night Round 4 ) and significant in other
# cases(for example: Tony Hawk's Underground and pro evolution soccer 2011)

# **•	Take a look at the general distribution of games by genre. What can we say about the most profitable genres? Can you generalize about genres with high and low sales?**

# In[ ]:


ax= df_games.pivot_table(index=['genre'],values='name',aggfunc='count').plot(kind='bar', figsize=(10,8), color='#0f9fd0');
plt.title('Total games in each genre');
plt.ylabel('Total numbers');
#genre.head(30)


# >Action and sport are popular genre

# In[ ]:


sns.set()
#df_games.pivot_table(index=['genre'],values='total_sales',aggfunc='sum').sort_values(by='total_sales', ascending=False).plot( figsize=(15,10));
pivot_genre_sales=pd.pivot_table(df_new, index=['year_of_release'],values='total_sales', columns='genre', aggfunc='sum')
plt.figure(figsize= (17,8))
sns.heatmap(pivot_genre_sales, cmap='RdBu_r');
plt.title('Heatmap showing distribution of different genre', fontdict={'size':20});
pivot_genre_sales.head()


# > There are significant amount of sales made by 'action'  over the period of 2000 to 2016. In the case of sports and misc genre, 
# in between 2007 to 2010 more than 75 million revenue  were collected every year by games of each genre.
# in between 2006 to 2010, most of the year either in sport or action genre or both about 140 million was collected per genre per
# year. 'Platform', 'Simuation' and 'strategy' genre seems to make less revenue ( mostly below 25 million).
# 

# **Conclusion**

# Data after year 2000 are significant.
# 
# PS2 is by far the most popular platform.
# 
# User_score has negligible effect on sales of the game. Critic score seem to have some influence in total_sales.
# 
# 'Action' and 'Sports' are all time most profit making genre.

# # Step 4. Create a user profile for each region

# In[ ]:


df_na= df_games[['platform','na_sales']].groupby(['platform']).sum().sort_values('na_sales', ascending=False).reset_index()
total_na=df_na['na_sales'].sum()
df_na['na_proportion']=df_na['na_sales']/total_na
df_na1=(df_na.set_index('platform')
        .replace('\$\s+','', regex=True)
        .astype(float)
        .applymap('{:,.3f}'.format))

na=df_na.head()
na1=df_na1.head()
na1


# In[ ]:


df_eu= df_games[['platform','eu_sales']].groupby(['platform']).sum().sort_values('eu_sales', ascending=False).reset_index()
total_eu=df_eu['eu_sales'].sum()
df_eu['eu_proportion']=df_eu['eu_sales']/total_eu
df_eu1=(df_eu.set_index('platform')
        .replace('\$\s+','', regex=True)
        .astype(float)
        .applymap('{:,.3f}'.format))
eu=df_eu.head()
eu1=df_eu1.head()
eu1


# In[ ]:


df_jp= df_games[['platform','jp_sales']].groupby(['platform']).sum().sort_values('jp_sales', ascending=False).reset_index()


# In[ ]:


total_jp=df_jp['jp_sales'].sum()
df_jp['jp_proportion']=df_jp['jp_sales']/total_jp
df_jp1=(df_jp.set_index('platform')
        .replace('\$\s+','', regex=True)
        .astype(float)
        .applymap('{:,.3f}'.format))
jp1=df_jp1.head()
jp=df_jp.head()
jp1


# In[ ]:


total1= pd.merge(left = na , right = eu, how='outer',on=['platform']).fillna(0)
total2=pd.merge(left = total1 , right = jp, how='outer',on=['platform']).fillna(0)
#total2.plot()

total22=(total2.set_index('platform')
        .replace('\$\s+','', regex=True)
        .astype(float)
        .applymap('{:,.3f}'.format))
total2.plot()

plt.title('Top 5  popular platforms in different regions');
plt.ylabel('total sales (milions)');
total22


# _X360, PS2 and DS_ make by far highest proportion of market in north america, europe and japan respectively making up  14%, 14%
# and 13.6%, correspondingly.
# In eu and na  all of _PS2, PS3,  PS4, Wii, X360_ make 9 to 14 % of market proportion in north maerica nad europe whereas later 
# three do not have any share
# in japanese market. _DS_ has no market share in north america but has highest market share in japan, likewise _PS_ (10.8%) which has 
# second highest market share in japnese market has no market share in europe.
# Likewise, *SNES* (9%) and *3DS* (7.8%) have market share in japan only.
# 
# But _PS2_ has substantial market share in all three market.
# 

# In[ ]:


df_na_genre= df_games[['genre','na_sales']].groupby(['genre']).sum().sort_values('na_sales', ascending=False).reset_index()
total_na_genre=df_na_genre['na_sales'].sum()
df_na_genre['na_proportion']=df_na_genre['na_sales']/total_na_genre
na_genre=df_na_genre.head()

df_eu_genre= df_games[['genre','eu_sales']].groupby(['genre']).sum().sort_values('eu_sales', ascending=False).reset_index()
total_eu_genre=df_eu_genre['eu_sales'].sum()
df_eu_genre['eu_proportion']=df_eu_genre['eu_sales']/total_eu_genre
eu_genre=df_eu_genre.head()

df_jp_genre= df_games[['genre','jp_sales']].groupby(['genre']).sum().sort_values('jp_sales', ascending=False).reset_index()
total_jp_genre=df_jp_genre['jp_sales'].sum()
df_jp_genre['jp_proportion']=df_jp_genre['jp_sales']/total_jp_genre
jp_genre=df_jp_genre.head()

total1_genre= pd.merge(left = na_genre , right = eu_genre, how='outer',on=['genre']).fillna(0)
total2_genre=pd.merge(left = total1_genre , right = jp_genre, how='outer',on=['genre']).fillna(0)
#total2.plot()

total22_genre=(total2_genre.set_index('genre')
        .replace('\$\s+','', regex=True)
        .astype(float)
        .applymap('{:,.3f}'.format))
total22_genre


# _Action and Sports_ are the most popular genra in all market. _Shooter_ on the other hand top 3 popular genra in european and north american market but has no market share in japan._platform_ genra has no share in european market but it stood at fourth position in other two regions. _Racing_ seem to be exist in european market , like wise _role-playing_ is popular only in japanese market.

# In[ ]:


df_games=df_games.drop_duplicates()


# In[ ]:


df_games.info()


# In[ ]:


df_games=df_games[df_games.rating.notnull()]
df_games.sample(10)


# In[ ]:


df_games.corr()


# In[ ]:


df_esrb=pd.pivot_table(df_games.sort_values(by='total_sales', ascending= False).head(20), index='platform', columns= ['rating'],values=['na_sales','eu_sales','jp_sales'], aggfunc='sum' ).fillna(0)
#.sort_values(by='total_sales', ascending= False)
df_esrb


# From pivot table, having '*E*' rating indicates higher sales in all three regions which is followed by '*M*' in  north america and by '*T*' in other two regions.

# ## Conclusion

# > X360, PS2 and DS  are most popular platform in all regions and other platforms popularity varies from 
# one region to another as explained earlier
# 
# >Action and sports  are most popular genre in all regions and other genres popularity varies from 
# one region to another as explained earlier
# 
# >having 'E' rating indicates higher sales in all three regions which is followed by 'M' in north america and by
# 'T' in other two regions.

# <div class="alert alert-success">
# <b>Reviewer's comment:</b> Well done, thank you for this sub-research.
# </div>
# 

# # Step 5. Test the following hypotheses:

# In[ ]:


df_games.info()


# In[ ]:


df_games['platform'].unique()


# ### Hypothesis

# Siginficance level is choosen 5% from three commonly used significance level(1%, 5% and 10%). 1% is ised for for critical testing
# like testing related to medicine , for example. On the other hand choosing 10 % could not give precise result in some cases.
# 
# > Null Hypothesis: Average user ratings of the Xbox One and PC platforms are the same.
# 
# > Alternative Hypothesis: Average user ratings of the Xbox One and PC platforms are not the same.

# In[ ]:


df_new.info()


# In[ ]:


df_new=df_new.dropna()
df_new.info()


# In[ ]:


rating=df_new[df_new['platform'].isin(['XOne',  'PC' ])]


# In[ ]:


rating_xone=rating[rating['platform'].isin(['XOne'])]
rating_pc=rating[rating['platform'].isin(['PC'])]


# In[ ]:


print(rating.groupby('platform')['user_score'].mean())


# In[ ]:


alpha = .05 # critical statistical significance level
                        # if the p-value is less than alpha, we reject the hypothesis
results_platform = st.ttest_ind(
    rating_xone['user_score'],         
    rating_pc['user_score'])

print('p-value: ', results_platform.pvalue)

if (results_platform.pvalue < alpha):
        print("We reject the null hypothesis")
else:
        print("We can't reject the null hypothesis")


# In[ ]:


genre_rating=df_new[df_new['genre'].isin(['Action',  'Sports' ])]
rating_action=genre_rating[genre_rating['genre'].isin(['Action'])]
rating_sports=genre_rating[genre_rating['genre'].isin(['Sports'])]
print(genre_rating.groupby('genre')[['user_score']].mean())


# ### Hypothesis

# >>Null Hypothesis: Average user ratings of the action and sports genre are the same.
# 
# >>Alternative Hypothesis: Average user ratings of the action and sports genre are not the same.

# In[ ]:





# In[ ]:


alpha = .05 # critical statistical significance level
                        # if the p-value is less than alpha, we reject the hypothesis
results_genre = st.ttest_ind(
    rating_action['user_score'],         
    rating_sports['user_score'])

print('p-value: ', results_genre.pvalue)

if (results_genre.pvalue < alpha):
        print("We reject the null hypothesis")
else:
        print("We can't reject the null hypothesis")


# **Conculsion**

# There is difference in user rating between 'PC' and 'Xone' platform but no significant difference between user rating's of given genres (Action and sports). 

# # Overal Conclusion

# >column labels are replace with lowercase, chnage data typed, and missing values are filled
# 
# >Sales in different platform and genre are analyzed. correlation of sales with user score and critic score is checked.
# 
# > Popular platform and sales distribution on this platfrom is visualized
# 
# > Rise and fadeout of platform is visualized
# 
# >there is  difference in user rating between given platforms
# 
# >no significant difference in user rating between given genre is obtained

#  To sum up, diffrent games and platform's popularity seem to be dependent on the region. However, platform: 'PS2'and genres :
# 'Action' and 'Sports'seems to be have most popular platform and genres in all regions. PS2_ platform has highest total sales at
# 1255.77 millions.Platform PS2 genrally made more money in north america follwed by european region. The maximum market share of 
# america, europe and japan were taken by X360, PS2 and Ds, respe In 2008,2009  and 2010 highest number of games 
# were released. 'Sports' and 'Action'genra's game made huge amount of money (up to about 140 million per year) at that time. 
# Also, other 
# games were making more amount of money  on these years than in other time. But, none of the platform was making drastic 
# improvement in sales on these specific years compared to sales in previous year ( see: Heatmap showing changes in  toal_sales
# in different platforms). Likewise, E rated games have higher sales irrespective of region followed by 'M' and 'T' rated games in 'north america' and other two regions respectively.
# 
# Likewise, the user rating for platform 'PC' found to be greater than for the platform 'Xone'. However, user rating for 'action' 
# genre and 'sports' genre has no significant difference. But, user_score and total sales seems to have negligible co-rellation. 
# On the other hand, critic_score seems to have some co-reelation (about 0.33) with total sales.
# 
# 

# In[ ]:




