
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 1
# 
# In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data. 
# 
# Each line of the `dates.txt` file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.
# 
# The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates. 
# 
# Here is a list of some of the variants you might encounter in this dataset:
# * 04/20/2009; 04/20/09; 4/20/09; 4/3/09
# * Mar-20-2009; Mar 20, 2009; March 20, 2009;  Mar. 20, 2009; Mar 20 2009;
# * 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
# * Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
# * Feb 2009; Sep 2009; Oct 2010
# * 6/2008; 12/2009
# * 2009; 2010
# 
# Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:
# * Assume all dates in xx/xx/xx format are mm/dd/yy
# * Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
# * If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
# * If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
# * Watch out for potential typos as this is a raw, real-life derived dataset.
# 
# With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.
# 
# For example if the original series was this:
# 
#     0    1999
#     1    2010
#     2    1978
#     3    2015
#     4    1985
# 
# Your function should return this:
# 
#     0    2
#     1    4
#     2    0
#     3    1
#     4    3
# 
# Your score will be calculated using [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), a correlation measure for ordinal data.
# 
# *This function should return a Series of length 500 and dtype int.*

# In[1]:


import pandas as pd

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)
df.head(10)


# In[4]:


import re

search1 = dict()
for ind,vals in dict(df.apply(lambda x:re.search('\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',x))).items():
    if vals:
        search1[ind]=vals.group()


# In[7]:


search1


# In[9]:



# Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
search2 = dict()
for ind,vals in dict(df.apply(lambda x:re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-zA-Z.,-]*[\s-]?(\d{1,2})?[,\s-]?[\s]?\d{4}',
                                                 x,re.I|re.M))).items():
    if vals and (ind not in list(search1.keys())):
        search2[ind]=vals.group()


# In[10]:


search2


# In[11]:



# 6/2008; 12/2009
search3 = dict()
for ind,vals in dict(df.apply(lambda x:re.search(r'\d{1,2}[/-]\d{4}',x,re.M|re.I))).items():
    if vals and (ind not in (list(search1.keys()) + list(search2.keys()))):

        search3[ind]=vals.group()


# In[15]:


# 2009; 2010
search4 = dict()
for ind,vals in dict(df.apply(lambda x:re.search('\d{4}',x,re.M|re.I))).items():
    if vals and (ind not in (list(search1.keys()) + list(search2.keys()) + list(search3.keys()) )):
        search4[ind] = vals.group()


# In[16]:


date_series = pd.concat([pd.Series(search1),pd.Series(search2),pd.Series(search3),pd.Series(search4)])


# In[18]:


date_series.head()


# In[23]:


def year_xx_to_xxxx(date):
  #search for date whose year is been encoded in two digits and make it four digits
    new_date = re.search(r'\d{1,2}[/]\d{1,2}[/]\d{2}\b',date)
    if pd.notnull(new_date):
        new_date = new_date.group()
        return new_date[:-2]+'19'+new_date[-2:]
    else:
        return date
year_xx_to_xxxx('23/2/19')


# In[37]:


new_date = '23/2/19'
new_date[:-2]+'19'+new_date[-2:]


# In[47]:


#If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009). Let's do this:
def insert_day(date):
#     search for dates with missing days and then add the day
    new_date = re.match(r'\d{1}[/]\d{4}\b',date)
    if pd.notnull(new_date):
        new_date = new_date.group()
        return new_date[:2]+'01/'+new_date[2:]
    else:
        return date
    
def insert_day2(date):
    new_date = re.match(r'\d{2}[/]\d{4}\b',date)
    if pd.notnull(new_date):
        new_date = new_date.group()
        return new_date[:2]+'/01'+new_date[2:]
    else:
        return date


# In[38]:


new_date = '9/2009'
new_date[:2]+'01/'+new_date[2:]


# In[46]:


def insert_month_day(date):
#     search for dates with only the year available
    new_date = re.match(r'\d{4}',date)
    if pd.notnull(new_date):
        new_date = new_date.group()
#         add day and month
        return '01/01/'+new_date
    else:
        return date


# In[19]:


def date_sorter():    
    # 04/20/2009; 04/20/09; 4/20/09; 4/3/09
    search1 = dict()
    for ind,vals in dict(df.apply(lambda x:re.search('\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',x))).items():
        if vals:
            search1[ind]=vals.group()

    # Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
    search2 = dict()
    for ind,vals in dict(df.apply(lambda x:re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-zA-Z.,-]*[\s-]?(\d{1,2})?[,\s-]?[\s]?\d{4}',
                                                     x,re.I|re.M))).items():
        if vals and (ind not in list(search1.keys())):
            search2[ind]=vals.group()

    # 6/2008; 12/2009
    search3 = dict()
    for ind,vals in dict(df.apply(lambda x:re.search(r'\d{1,2}[/-]\d{4}',x,re.M|re.I))).items():
        if vals and (ind not in (list(search1.keys()) + list(search2.keys()))):

            search3[ind]=vals.group()

    # 2009; 2010
    search4 = dict()
    for ind,vals in dict(df.apply(lambda x:re.search(r'\d{4}',x,re.M|re.I))).items():
        if vals and (ind not in (list(search1.keys()) + list(search2.keys()) + list(search3.keys()))):
            search4[ind]=vals.group()

    date_series = pd.concat([pd.Series(search1),pd.Series(search2),pd.Series(search3),pd.Series(search4)])
   


# In[59]:



def alph_to_digit(month):
    month_cld = ''
    if re.search(r'\bJan\w*',month,flags=re.I):
        month_cld = re.search(r'\bJan\w*',month,flags=re.I).group()
        return month.replace(month_cld,'1')
    elif re.search(r'\bfeb\w*',month,flags=re.I):
        month_cld = re.search(r'\bfeb\w*',month,flags=re.I).group()
        return month.replace(month_cld,'2')
    elif re.search(r'\bmar\w*',month,flags=re.I):
        month_cld = re.search(r'\bmar\w*',month,flags=re.I).group()
        return month.replace(month_cld,'3')
    elif re.search(r'\bapr\w*',month,flags=re.I):
        month_cld = re.search(r'\bapr\w*',month,flags=re.I).group()
        return month.replace(month_cld,'4')
    elif re.search(r'\bmay\w*',month,flags=re.I):
        month_cld = re.search(r'\bmay\w*',month,flags=re.I).group()
        return month.replace(month_cld,'5')
    elif re.search(r'\bjun\w*',month,flags=re.I):
        month_cld = re.search(r'\bjun\w*',month,flags=re.I).group()
        return month.replace(month_cld,'6')
    elif re.search(r'\bjul\w*',month,flags=re.I):
        month_cld = re.search(r'\bjul\w*',month,flags=re.I).group()
        return month.replace(month_cld,'7')
    elif re.search(r'\baug\w*',month,flags=re.I):
        month_cld = re.search(r'\baug\w*',month,flags=re.I).group()
        return month.replace(month_cld,'8')
    elif re.search(r'\bsep\w*',month,flags=re.I):
        month_cld = re.search(r'\bsep\w*',month,flags=re.I).group()
        return month.replace(month_cld,'9')
    elif re.search(r'\boct\w*',month,flags=re.I):
        month_cld = re.search(r'\boct\w*',month,flags=re.I).group()
        return month.replace(month_cld,'10')
    elif re.search(r'\bnov\w*',month,flags=re.I):
        month_cld = re.search(r'\bnov\w*',month,flags=re.I).group()
        return month.replace(month_cld,'11')
    elif re.search(r'\bdec\w*',month,flags=re.I):
        month_cld = re.search(r'\bdec\w*',month,flags=re.I).group()
        return month.replace(month_cld,'12')
    else:
        return month


def year_xx_to_xxxx(date):
    new_date = re.search(r'\d{1,2}[/]\d{1,2}[/]\d{2}\b',date)
    if pd.notnull(new_date):
        new_date = new_date.group()
        return new_date[:-2]+'19'+new_date[-2:]
    else:
        return date
    

def insert_day(date):
#     search for dates with missing days and then add the day
    new_date = re.match(r'\d{1}[/]\d{4}\b',date)
    if pd.notnull(new_date):
        new_date = new_date.group()
        return new_date[:2]+'01/'+new_date[2:]
    else:
        return date
    

def insert_day2(date):
    new_date = re.match(r'\d{2}[/]\d{4}\b',date)
    if pd.notnull(new_date):
        new_date = new_date.group()
        return new_date[:2]+'/01'+new_date[2:]
    else:
        return date
    

def insert_month_day(date):
#     search for dates with only the year available
    new_date = re.match(r'\d{4}',date)
    if pd.notnull(new_date):
        new_date = new_date.group()
#         add day and month
        return '01/01/'+new_date
    else:
        return date



def date_sorter():    
    # 04/20/2009; 04/20/09; 4/20/09; 4/3/09
    search1 = dict()
    for ind,vals in dict(df.apply(lambda x:re.search('\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',x))).items():
        if vals:
            search1[ind]=vals.group()

    # Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
    search2 = dict()
    for ind,vals in dict(df.apply(lambda x:re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-zA-Z.,-]*[\s-]?(\d{1,2})?[,\s-]?[\s]?\d{4}',
                                                     x,re.I|re.M))).items():
        if vals and (ind not in list(search1.keys())):
            search2[ind]=vals.group()

    # 6/2008; 12/2009
    search3 = dict()
    for ind,vals in dict(df.apply(lambda x:re.search(r'\d{1,2}[/-]\d{4}',x,re.M|re.I))).items():
        if vals and (ind not in (list(search1.keys()) + list(search2.keys()))):

            search3[ind]=vals.group()

    # 2009; 2010
    search4 = dict()
    for ind,vals in dict(df.apply(lambda x:re.search(r'\d{4}',x,re.M|re.I))).items():
        if vals and (ind not in (list(search1.keys()) + list(search2.keys()) + list(search3.keys()))):
            search4[ind]=vals.group()

    date_series = pd.concat([pd.Series(search1),pd.Series(search2),pd.Series(search3),pd.Series(search4)])
    
#     return date_series

    date_c = date_series.apply(alph_to_digit)
    date_c = date_c.str.strip().str.replace('-','/')

    date_c = date_c.apply(lambda x: year_xx_to_xxxx(x))
    date_c = date_c.apply(lambda x:insert_day(x))
    date_c = date_c.apply(lambda x:insert_day2(x))
    date_c = date_c.apply(lambda x:insert_month_day(x))
#     return pd.Series(pd.to_datetime(date_c).sort_values().index)
    return pd.Series(pd.to_datetime(date_c).sort_values().index)


# In[60]:


date_sorter()

