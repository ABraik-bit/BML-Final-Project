import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from tqdm.notebook import tqdm
import collections
import pickle
import gc

# Read all files
json_files = []
csv_files = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        fname = os.path.join(dirname, filename)
        if fname.endswith('csv'):
            csv_files.append(fname)
        elif fname.endswith('json'):
            json_files.append(fname)

# Reorder CSV files
country_codes = list(map(lambda string: ''.join(list(filter(lambda word: word.isupper(), string))), csv_files))
country_codes, order = zip(*sorted(list(zip(country_codes, range(len(country_codes)))), key=lambda val: val[0]))
csv_files = [csv_files[ind] for ind in order]

# Reorder json files
country_codes = list(map(lambda string: ''.join(list(filter(lambda word: word.isupper(), string))), json_files))
country_codes, order = zip(*sorted(list(zip(country_codes, range(len(country_codes)))), key=lambda val: val[0]))
json_files = [json_files[ind] for ind in order]


def initialize_country_dataframe(dataframe, json_fname, country_code):
    '''First, remove duplicate rows from the dataframe, second, map category_id column to actual categories, thrid,
    new column in the dataframe called country_code'''

    df = dataframe.copy()
    df.drop_duplicates(inplace=True)

    with open(json_fname, 'r') as f:
        json_data = json.loads(f.read())

    mapping_dict = dict([(int(dictionary['id']), dictionary['snippet']['title']) for dictionary in json_data['items']])

    df['category'] = df['category_id'].replace(mapping_dict)
    del df['category_id']

    df['country_code'] = country_code

    return df


# Initialize country-by-country dataframe using above written function
dataframes = []
for ind, code in enumerate(country_codes):
    try:
        df = pd.read_csv(csv_files[ind])
    except:
        df = pd.read_csv(csv_files[ind], engine='python')

    df = initialize_country_dataframe(df, json_files[ind], code)
    print(code, df.shape)
    dataframes.append(df)

# Concatenate individual dataframe to form single main dataframe
dataframe = pd.concat(dataframes)
print(dataframe.shape)

# Remove videos with unknown video id
drop_index = dataframe[dataframe.video_id.isin(['#NAME?', '#VALUE!'])].index
dataframe.drop(drop_index, axis=0, inplace=True)

# Reading pre-calculated dictionaries
with open('/kaggle/input/id-dayspickle/id_days.pickle','rb') as f:
    id_days=pickle.load(f)

with open('/kaggle/input/id-countriespickle/id_countries.pickle','rb') as f:
    id_countries=pickle.load(f)

num_days=id_days.values()
num_countries=id_countries.values()

# Adding feature num_days into the dataframe
def n_days_replace(vid):
    return id_days[vid]

dataframe['num_days']=dataframe.video_id.apply(func=n_days_replace)

# Adding feature num_countries into the dataframe
def n_countries_replace(vid):
    return id_countries[vid]

dataframe['num_countries']=dataframe.video_id.apply(func=n_countries_replace)


def freq_of_days_video_in_trend():
    trending_days = collections.Counter(num_days)
    days, freq = zip(*sorted(trending_days.items(), key=lambda val: val[0]))

    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))

    cmap = plt.get_cmap('GnBu')
    colors = [cmap(i) for i in np.linspace(0, 1, len(days))]
    ax1.bar(range(len(days)), np.log(freq), color=colors)
    ax1.set_xticks(range(len(days)))
    ax1.set_xticklabels(days)

    labels = [str(val) for val in freq]
    for ind, val in enumerate(np.log(freq)):
        ax1.text(ind, val + 0.1, labels[ind], ha='center')

    ax1.set_xticks(range(len(days)))
    ax1.set_xticklabels(days)

    ax1.set_ylabel('Log frequency')

    cum_arr = np.cumsum(freq)
    max_val = np.max(cum_arr)
    min_val = np.min(cum_arr)

    ax2.plot((cum_arr - min_val) / (max_val - min_val))
    ax2.set_xticks(range(len(days)))
    ax2.set_xticklabels(days)
    ax2.set_ylabel('Cumulative proportion of number of videos')
    ax2.set_xlabel('For number of days videos are in trend');

def freq_of_num_of_countries_videos_in_trend():
    trending_countries = collections.Counter(num_countries)
    nc, freq = zip(*sorted(trending_countries.items(), key=lambda val: val[0]))

    fig, ax = plt.subplots(figsize=(14, 6))

    cmap = plt.get_cmap('PuBu')
    colors = [cmap(i) for i in np.linspace(0, 1, len(nc))]
    ax.bar(range(len(nc)), np.log(freq), color=colors)
    ax.set_xticks(range(len(nc)))
    ax.set_xticklabels(nc)

    labels = [str(val) for val in freq]
    for ind, val in enumerate(np.log(freq)):
        ax.text(ind, val + 0.1, labels[ind], ha='center')

    ax.set_ylabel('Log frequency')
    ax.set_xlabel('For number of countries videos are in trend')
    ax.set_title('Discrete log frequency plot for the number of countries videos are in trend');

#	Does number of videos published in a month vary over the months?
def q3():
    df = unique_video_id(keep='first')

    months, counts = zip(*sorted(df.publish_month.value_counts().to_dict().items(), key=lambda val: val[0]))

    fig, ax = plt.subplots(figsize=(15, 5))

    cmap = plt.get_cmap('Set3')
    colors = [cmap(i) for i in range(len(months))]

    ax.bar(months, counts, color=colors)
    ax.set_xticks(range(1, len(months) + 1))
    ax.set_xticklabels(months)
    ax.set_xlabel('Months')
    ax.set_ylabel('Number of videos published')
    for ind, val in enumerate(counts):
        ax.text(months[ind], val + 500, val, ha='center');

#4.	Do the trending videos from the data set are published in specific time slot of 24 hours day more than the other times?
def q4():
    df = unique_video_id(keep='first')

    hours, counts = zip(*sorted(df.publish_hour.value_counts().to_dict().items(), key=lambda val: val[0]))

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.get_cmap('twilight')
    colors = [cmap(i) for i in np.linspace(0, 1, len(hours))]

    ax.bar(hours, counts, color=colors)
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels(hours)
    ax.set_xlabel('Hour of a day')
    ax.set_ylabel('Number of videos published');

#5.	How long usually it takes for the videos to become trending?
def q5():
    df = unique_video_id(keep='first')

    days_lapse = df['days_lapse']
    days_lapse_count = days_lapse.value_counts().to_dict()
    days, count = zip(*sorted(list(filter(lambda val: val[1] > 1, days_lapse_count.items())), key=lambda val: val[0]))

    fig, [ax1, ax2] = plt.subplots(figsize=(19, 13), nrows=2, ncols=1)

    cmap = plt.get_cmap('autumn')
    colors = [cmap(i) for i in np.linspace(0, 1, len(days))]

    ax1.bar(range(len(days)), np.log(count), width=0.6, color=colors)
    ax1.set_xticks(range(len(days)))
    ax1.set_xticklabels(days, rotation=45)
    ax1.set_ylabel('log of frequency count')
    ax1.set_xlabel('Number of days pass before videos are trending')
    ax1.set_title('Discrete frequency plot for number of days before videos are trending')

    cum_arr = np.cumsum(count)
    max_val = np.max(cum_arr)
    min_val = np.min(cum_arr)
    ax2.plot((cum_arr - min_val) / (max_val - min_val))
    ax2.set_xticks(range(len(days)))
    ax2.set_xticklabels(days, rotation=45)
    ax2.set_ylabel('Cumulative proportion of number of videos')
    ax2.set_xlabel('Number of days pass before videos are trending');

#6.	Resolving the misconception of the effect of the hour’s videos are published.
def q6():
    df = unique_video_id(keep='first')
    print('Total number of unique videos:', df.shape[0])
    print('Total number of unique videos that take less than 31 days to be in trend:', df[df.days_lapse < 31].shape[0])
    fig, ax = plt.subplots(figsize=(20, 6))
    df[df.days_lapse < 31].boxplot(column='days_lapse', by='publish_hour', rot=90, ax=ax)
    ax.set_xlabel('Hours')
    ax.set_ylabel('days lapse')
    ax.set_title('')
    fig.suptitle('Box plot for videos that took less than 31 days to be in trend');

#7.	Percentage of videos by the countries that have comments section disabled.
def q7():
    df_disabled = unique_video_id()
    df_disabled = df_disabled[df_disabled.comments_disabled == True]
    df_disabled.sort_values(by=['video_id', 'comments_disabled'], inplace=True)
    df_disabled.drop_duplicates(subset='video_id', keep='last', inplace=True)
    disabled_dict = df_disabled.country_code.value_counts().to_dict()

    df_enabled = unique_video_id()
    df_enabled = df_enabled[df_enabled.comments_disabled == False]
    df_enabled.sort_values(by=['video_id', 'comments_disabled'], inplace=True)
    df_enabled.drop_duplicates(subset='video_id', keep='first', inplace=True)
    enabled_dict = df_enabled.country_code.value_counts().to_dict()

    dis_ena_prop = {}
    for country in disabled_dict.keys():
        dis_ena_prop[country] = disabled_dict[country] / enabled_dict[country]

    fig, ax = plt.subplots(figsize=(13, 6))

    cmap = plt.get_cmap('Set3')
    colors = [cmap(i) for i in range(len(days))]

    countries = dis_ena_prop.keys()
    values = list(dis_ena_prop.values())
    ax.bar(countries, np.log(np.array(list(values)) + 1), color=colors)
    ax.set_ylabel('log(1+values) transformed values')
    ax.set_xlabel('Country codes')
    ax.set_title('Percentage of unique trending videos that have comments section disabled over each country')

    for ind, val in enumerate(np.log(np.array(list(values)) + 1)):
        ax.text(ind, val + 0.001, str(round(values[ind] * 100, 1)) + '%', ha='center')

#8.	Number of views by the video categories
def q8():
    df = unique_video_id()

    # 1. Get number of views for each categories
    views_per_category = df.groupby(by=['category'], as_index=False).views.sum()

    # 2. Get number of views and category names
    views_in_million = [int(views / 1000000) for views in views_per_category.sort_values(by='views').views.values]
    cat_val = views_per_category.sort_values(by='views').category.values

    # 3. Normalize number of views for data visualization
    relative_vals = views_per_category.sort_values(by='views').views.values
    max_val = np.max(relative_vals)
    min_val = np.min(relative_vals)
    diff = max_val - min_val
    ms_val = (relative_vals - min_val) / diff

    # 4. Create axes for plotting
    fig, ax = plt.subplots(figsize=(20, 3))
    # 4.1 Add one more axis
    bx = ax.twiny()
    x = range(len(cat_val))
    y = [5] * len(cat_val)

    # 5. Plot one category at a time using matplotlib scatter plot
    for ind, cat in enumerate(cat_val):
        ax.scatter(x[ind], y[ind], s=ms_val[ind] * 10000, cmap='Blues', alpha=0.5, edgecolors="grey", linewidth=2)
    ax.set_xticks(range(len(cat_val)))
    ax.set_xticks(range(len(cat_val)))
    ax.set_yticklabels([])
    ax.set_xticklabels(cat_val, rotation=90)
    ax.set_xlabel('Video categories', fontsize=16)

    # 6. Write number of views in millions on the x-axis above the plot
    bx.set_xticks(range(len(views_in_million)))
    bx.set_xticklabels(views_in_million, rotation=90)
    bx.set_xlabel('Number of views in millions', fontsize=16);

#9.	Proportion of video categories trending in the countries over the entire period of given time.
def q9():
    # 1. Replace category 29 by string 'Other'
    dataframe.category.replace({29: 'Other'}, inplace=True)

    # 2. Count number of occurances of video category for each category
    country_by_category = dataframe.groupby(by='country_code')['category'].value_counts()

    # 3. Write function that will plot a pie-chart
    def pie_chart(country_code, axis):
        '''Plots a pie_chart for a country_by_category series for a given country code on given axis'''
        cmap = plt.get_cmap('Spectral')
        colors = [cmap(i) for i in np.linspace(0, 1, len(country_by_category[country_code].index))]
        axis.pie(country_by_category[country_code].values, labels=country_by_category[country_code].index,
                 autopct='%.2f', colors=colors, shadow=True)
        axis.set_title(country_code, fontsize=14);

    # 4. Plot individual pie-chart for each country
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(16, 20))
    for c_i in range(0, len(country_codes), 2):
        col = 0
        ind = c_i // 2
        pie_chart(country_codes[c_i], ax[ind][col])
        pie_chart(country_codes[c_i + 1], ax[ind][col + 1])

    # 5. Plot pie-chart for all countries together
    fig, ax = plt.subplots(figsize=(4, 4))
    cmap = plt.get_cmap('Spectral')
    all_countries_prop = dataframe.category.value_counts()
    colors = [cmap(i) for i in np.linspace(0, 1, len(all_countries_prop.index))]
    ax.pie(all_countries_prop.values, labels=all_countries_prop.index, autopct='%.2f', colors=colors, shadow=True)
    ax.set_title('All Countries');

def ML_Model():
    # Create train and test data frames for prediction
    X_train = train[
        ['comments_disabled', 'ratings_disabled', 'video_error_or_removed', 'category', 'country_code', 'num_countries',
         'num_days', 'days_lapse', 'views_cat', 'comment_count_cat']]
    X_test = test[
        ['comments_disabled', 'ratings_disabled', 'video_error_or_removed', 'category', 'country_code', 'num_countries',
         'num_days', 'days_lapse', 'views_cat', 'comment_count_cat']]

    train_rows = X_train.shape[0]
    data = pd.concat([X_train, X_test])

    data = pd.get_dummies(data)

    X_train = data[:train_rows].copy()
    X_test = data[train_rows:].copy()

    del data
    gc.collect()

    X_train.shape, X_test.shape

    # Baseline linear regression model
    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(X_train, np.log(y_train + 1))
    lr.score(X_train, np.log(y_train + 1))

    from sklearn import metrics

    y_pred = lr.predict(X_test)
    y_pred = np.exp(y_pred) - 1
    metrics.mean_absolute_error(y_test, y_pred)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #run the function needed

