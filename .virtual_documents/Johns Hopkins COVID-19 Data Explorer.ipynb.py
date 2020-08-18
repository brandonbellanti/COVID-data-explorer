# load libraries
import pandas as pd, numpy as np, matplotlib.pyplot as plt,seaborn as sns,math,warnings
get_ipython().run_line_magic("matplotlib", " inline")
get_ipython().run_line_magic("config", " InlineBackend.figure_format = 'retina'")
warnings.filterwarnings('ignore')

# urls for jhu datasets
cases_csv_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
deaths_csv_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'

# load dataframes from urls
cases_df = pd.read_csv(cases_csv_url)
deaths_df = pd.read_csv(deaths_csv_url)

# combine dfs with new column for report type
cases_df['Report_Type'] = 'cases'
deaths_df['Report_Type'] = 'deaths'
reports_df = cases_df.append(deaths_df,ignore_index=True)

# rename date columns (pad month and day values) for later filtering/sorting
date_cols = reports_df.filter(regex=r'\d{1,2}\/\d{1,2}\/\d{2}').columns.to_list()
date_formatter = lambda x: "{:0>2}/{:0>2}/{:0>2}".format(*(x.split('/')))
date_cols_map = {k: date_formatter(k) for k in date_cols}
reports_df = reports_df.rename(columns=date_cols_map)
date_cols = list(date_cols_map.values())
today = date_cols[-1]


# list US states and District of Columbia for later filtering/sorting
us_states = reports_df[(reports_df['iso3']=='USA')& \
            (reports_df['Admin2'].notna())]['Province_State'].unique()

# create df for US states data
states_df = reports_df[reports_df['Province_State'].isin(us_states)] \
            .set_index(['Report_Type','Province_State','Admin2'])[date_cols]


# plot an average line

report_type = 'cases'
state = 'Michigan'
days = 14
week_dates = date_cols[(-days-1):-1]
trend_df = states_df.query(f'Report_Type=="{report_type}" & Province_State=="{state}"').groupby(level='Province_State').sum().diff(axis=1)[week_dates]
# trend_df = trend_df[week_dates].where(~(trend_df[week_dates]<0), other=np.nan)
trend_df = trend_df.transpose().reset_index().rename(columns={'index':'Date'})
print(trend_df)
x = trend_df.index
y = trend_df[state]

plt.bar(x,y)
plt.xticks(ticks=np.arange(0,len(week_dates),2),labels=week_dates[::2],rotation=45,ha='right')
# trend line plot code from https://widu.tumblr.com/post/43624347354/matplotlib-trendline
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),'r')


# total us cases/deaths
states_df.groupby(level='Report_Type')[today].sum().to_frame(name='US_Totals')


# change in us cases/deaths in previous 7 days
states_df.groupby(level='Report_Type').sum().diff(axis=1)[date_cols[-8:-1]].sum(axis=1).to_frame(name='US_Weekly_Change')


# most recent case/death count by state
report_type = 'deaths'
states_df.loc[report_type].groupby(level='Province_State')[date_cols[-1]] \
            .sum().sort_values(ascending=False).to_frame(name='Cumulative_Cases').head(20)


# new cases/deaths in the last 7 days
report_type = 'cases'
states_df.loc[report_type].groupby(level='Province_State').sum().diff(axis=1)[date_cols[-7:]].sum(axis=1) \
            .sort_values(ascending=False).to_frame(name='Weekly_Change').head(10)


# cases in the last 7 days
states_df.loc['cases'].groupby('Province_State').sum()[[date_cols[-8],today]].diff(axis=1)[date_cols[-1]]#.sort_values(ascending=False).head()


# cases in the last 7 days
weekly_change_df = states_df.loc['cases'].groupby('Province_State').sum().diff(axis=1,periods=7)[today].to_frame(name='new_cases')
weekly_change_df['total_cases'] = states_df.groupby('Province_State').sum()[today]
weekly_change_df['pct_change'] = weekly_change_df['new_cases']/weekly_change_df['total_cases']

cm = sns.light_palette("red", as_cmap=True)
weekly_change_df.sort_values(by='pct_change', ascending=False).style.background_gradient(cmap=cm)


# calculate totals, change, and percent change based on weekly interval
interval= 7
start = (len(date_cols) % interval) - 1
weekly_change_df = states_df.loc['cases'].groupby('Province_State').sum().diff(axis=1,periods=7)[date_cols[start::interval]] # weekly change ending on date
weekly_totals_df = states_df.loc['cases'].groupby('Province_State').sum()[date_cols[start::interval]] # case totals ending on date
weekly_change_pct_df = weekly_change_df/weekly_totals_df # percent of weekly change over total cases

cm = sns.diverging_palette(240, 5, as_cmap=True)
weekly_change_pct_df.diff(axis=1)[weekly_change_pct_df.columns[-10:]].style.background_gradient(cmap=cm).highlight_null('white') # change in weekly percentage from week to week


def plot_county_heatmaps(state,report_type='cases'):
    test_df = states_df.loc[(report_type,state)][date_cols].sort_values(by=[date_cols[-1]],ascending=False).diff(axis=1).fillna(0).head(8)
    test_df = test_df[date_cols].where(~(test_df[date_cols]<0), other=np.nan)
    counties = test_df.index.to_list()

    fig, ax = plt.subplots(len(counties),1,figsize=(14, len(counties)/1.5))
    for i,county in enumerate(counties):
        nbins = 4
        bin_labels = [i for i in range(nbins)]
        plot = pd.cut(test_df.loc[county],bins=nbins,labels=bin_labels).astype('float').to_frame(name='cases').transpose()
        sns.heatmap(plot,cmap='Reds',cbar=False,yticklabels=[county],xticklabels=False,ax=ax[i])
        plt.setp(ax[i].yaxis.get_majorticklabels(), rotation=0)

    plt.show()
    
plot_county_heatmaps('Massachusetts')


# get cumulateive case and death counts

def cumulative_case_and_death_counts(state=False,county=False,date=date_cols[-1]):
    """Returns the total case and death counts for
    a given county and state as of a given date.
    If no state or county is passed, the US totals will be used
    If a date is not passed, the most recent date will be used."""
    
    if county:
        counts = states_df.query(f'Province_State=="{state}" & Admin2=="{county}"')[date]
    elif state:
        counts = states_df.query(f'Province_State=="{state}"').groupby(['Report_Type','Province_State']).sum()[date]
    else:
        counts = states_df.groupby('Report_Type').sum()[date].to_frame()
    
    res = f"""There have been {counts.loc['cases'][0]} cases of COVID-19 and """ \
            f"""{counts.loc['deaths'][0]} deaths reported in {f'{county} County, ' if county else ''}""" \
            f"""{state if state else 'the United States'} as of {date}."""
    
    return res


# get first reported case date and return dates for plotting
def get_plot_dates(state=False):
    first_reported_grouping = ('cases',state) if state else ('cases')
    first_reported_case = states_df.loc[first_reported_grouping].sum().ne(0).idxmax()
    plot_dates = date_cols[date_cols.index(first_reported_case)-1:] if state else date_cols
    return plot_dates

# fix error for US data


# plot daily change in cases/deaths for a given state or county

def plot_daily_change(state=False,county=False,report_type='cases',time_frame='all'):
    """Plots the daily change in cases/deaths for a given state or county.
If no state or county is passed, the total US count will be used.

Accepted `report_type` values are:
 - 'cases': case data
 - 'deaths': death data
    
"""
    plot_dates = get_plot_dates(state)
    
    if county:
        plot_df = states_df.loc[(report_type,state,county)].diff()[plot_dates]
    elif state:
        plot_df = states_df.loc[(report_type,state)].sum().diff()[plot_dates]
    else:
        plot_df = states_df.loc[report_type].sum().diff()[plot_dates]

    plt.figure(figsize=(12, 5))
    
    color = 'tab:red' if report_type=='cases' else 'tab:gray'
    
    daily_cases = plot_df.plot(kind='bar',color=color,alpha=.4)
    
    average_cases = plot_df.rolling(7,center=True).mean()
    average_cases.plot(linewidth=2,color=color)
    
    plt.xticks(ticks=np.arange(0,len(plot_dates), 15),labels=plot_dates[::15], \
                rotation=45,ha='right')
    
    plt.title(f"""Seven-day rolling average of daily COVID-19 {report_type} for """ \
              f"""{f'{county} County, ' if county else ''}""" \
              f"""{state if state else 'the United States'} as of {date_cols[-1]}""")
    
    plt.ylabel(f'Number of {report_type}')
    plt.xlabel('Date')
    plt.legend().remove()
    
    for i in list(plt.yticks()[0][::2]):
        plt.axhline(y=i,linewidth=1.25, color='lightgray',alpha=.75, \
                    linestyle='dashed',zorder=-5)
   
    plt.fill_between(plot_dates,average_cases,color=color,alpha=.15)
    plt.show()


# view a report of cases/deaths with charts for a given state or county

def status_report(state=False,county=False):
    
    print(f"\n{cumulative_case_and_death_counts(state,county)}\n")
    
    for report_type in ['cases','deaths']:
        plot_daily_change(state,county,report_type=report_type)


# view report for United States

status_report()


# view report for Massachusetts

status_report(state='Massachusetts')


# view report for New York City, New York

status_report(state="Michigan", county="Oakland")
