# Analyzing_Streaming_Content_Data

https://github.com/jadvanishraddha/Analyzing_Streaming_Content_Dataset/tree/main
## INTRODUCTION

In the digital era, streaming platforms such as Netflix have transformed the way people consume entertainment. With a vast collection of movies, TV shows, and documentaries available globally, understanding the content catalog can help stakeholders make data-driven decisions about content acquisition, user engagement, and regional preferences. This project involves analyzing a dataset that contains detailed information about streaming content, including titles, directors, cast, countries, genres, release year, and more.



## PROBLEM STATMENT

The main challenge is to extract valuable insights from the streaming content dataset. Some key questions that guide the analysis include:

What type of content (TV Shows or Movies) dominates the platform?

Which countries contribute the most content?

What is the distribution of content over the years?

Who are the most frequent directors and actors?

Which genres are most popular?

By addressing these questions, we aim to understand content trends and user preferences, which can ultimately help improve content strategy and user satisfaction.



## PROJECT OUTCOME

By the end of this project, we aim to:

Clean and preprocess the dataset for analysis.

Visualize content trends by type, country, release year, and rating.

Identify top contributing directors, cast members, and countries.

Determine patterns in content duration and genre categorization.

Provide actionable insights that can help with content recommendation, acquisition, and platform strategy.

## Importing libraries required for this data set
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport


## Loading the dataset
df = pd.read_csv(r"C:\Users\jadva\AppData\Local\Temp\Rar$DIa12128.49053\netflix_titles.csv")
df.columns

## Getting information about the dataset
df.info()

## Looking at the number of rows and columns in the dataset
df.shape

## Having glance at the first five rows of dataset
df.head()

## Having glance at the last five rows of dataset
df.tail(10)

## To find is there any null value in the dataset
df.isnull().sum()

## Fill The Null Value With Mode
df["director"] = df["director"].fillna(df["director"].mode()[0])
df["cast"] = df["cast"].fillna(df["cast"].mode()[0])
df["country"] = df["country"].fillna(df["country"].mode()[0])
df["date_added"] = df["date_added"].fillna(df["date_added"].mode()[0])
df["rating"] = df["rating"].fillna(df["rating"].mode()[0])
df["duration"] = df["duration"].fillna(df["duration"].mode()[0])

## To Check If There Is Any Null Value 
df.isnull().sum()

## To find is there any duplicate value in dataset
df.corr(numeric_only=True)

## Data Visualization

### Bar Plot Showing Pollutant Distribution Across State
sns.barplot(data=df, x='type', y='rating')
plt.title('rating analysis ')
plt.xticks(rotation=90)
plt.show()

## analysis the highest release
plt.figure(figsize=(14,6))
sns.boxplot(data=df, x='rating', y='release_year')
plt.title('analysis the highest release')
plt.show()

## tvshow and movies data
sns.scatterplot(data=df, x='release_year', y='rating', hue='type')
plt.title('tv show and movies data')
plt.show()

## Most Frequently Recorded 
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='rating', order=df['rating'].value_counts().index)
plt.title('Frequency of rating ')
plt.show()

## Pie Chart of movie and tvshow
pollutant_data = df.groupby('type')['release_year'].sum()

plt.figure(figsize=(6, 6))
plt.pie(pollutant_data, labels=pollutant_data.index, autopct='%1.1f%%', startangle=140)
plt.title('release data of movie and tvshow')
plt.axis('equal')
plt.show()

## Correlation
df_numeric = df.select_dtypes(include=['number'])
plt.figure(figsize=(12,6))
sns.heatmap(df_numeric.corr(), annot=True, cmap='magma')
plt.title("")
plt.show()

## Pair Plot
sns.pairplot(df, diag_kind='kde')
plt.suptitle('Pair Plot release year',  y=1.02)
plt.show()

## Creating Report Using Y-Data Profiling
profile = ProfileReport(df, title='Netflix Report', explorative=True)
profile.to_notebook_iframe()

## CONCLUSION

Through this project, we successfully analyzed key trends and patterns within the streaming content dataset. We observed notable preferences in content type, release trends over the years, and dominant genres across regions. The findings not only help understand the content library better but also set a strong foundation for building recommendation systems, user preference models, or strategic content curation for streaming platforms. The project demonstrates how data analytics can offer significant value in the entertainment and media industry.
