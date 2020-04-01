import argparse
import urllib.request
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
plt.style.use('default')
plt.rcParams['font.sans-serif'] = "Arial"


def get_data():
    # Name URLs
    confirmed_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    recovered_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
    death_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

    # Create a dictionary of file name and url name
    file_names = ['data/confirmed.csv', 'data/recovere.csv', 'data/death.csv']
    urls = [confirmed_url, recovered_url, death_url]
    case_types = ['confirmed', 'recovered', 'death']
    data = {}

    # Create data/ folder
    Path('data/').mkdir(exist_ok=True)

    for file_name, url, case_type in zip(file_names, urls, case_types):
        # Save csv files from url
        urllib.request.urlretrieve(url, file_name)

        # Read csv to pd
        data[case_type] = pd.read_csv(file_name)

        # lowercase column names
        data[case_type].columns = map(str.lower, data[case_type].columns)

        # Date strings to datetime object
        column = ['province/state', 'country/region', 'lat', 'long']
        for i in data[case_type].columns[4:]:
            column.append(datetime.strptime(i, '%m/%d/%y'))
        data[case_type].columns = column
    return data


def get_dates(data, case_type):
    """
    takes the column headers related to dates
    returns the dates
    """
    return np.asarray(data[case_type].columns[4:])


def get_num_cases(data, case_type, country, province=None):
    """
    return the number of cases by case type and geography
    """
    if province is not None:
        # number of cases for a province
        condition = ((data[case_type]['country/region'] == country) &
                     (data[case_type]['province/state'] == province))
        result = data[case_type][condition].iloc[:, 4:].values.flatten()
    else:
        # number of cases for the full country/region
        result = data[case_type][
            data[case_type]['country/region'] == country
        ].iloc[:, 4:].sum(axis=0).values.flatten()
    return result


def get_end_month(month):
    """
    takes in month
    returns the date at the end of that month
    """
    return datetime(2020, month, 28)


def get_end_months(dates):
    """
    takes in the dates
    return the end of month corresponding to the dates
    """

    # Turn dates array into list
    date_list = dates.tolist()

    # List of all the months
    month_list = list(set([date.month for date in date_list]))

    # Compile all the end of months into a list
    end_months = []
    for month in month_list:
        end_months.append(get_end_month(month))

    return end_months


def format_datetime(dt):
    """
    format as mm-yyyy
    """
    return dt.strftime('%m-%Y')


def get_rgb(color):
    """
    normalizes RGB values
    """
    r, g, b = color
    color = (r / 255., g / 255., b / 255.)
    return color


def get_title(country, province=None):
    """
    """
    if province is None:
        return country
    else:
        return province + ' (' + country + ')'


def plot_case_by_country(data, country, province):
    """
    plot main case types - confirmed, recovered, death
    """
    case_types = ['confirmed', 'recovered', 'death']
    # Plot cases by country
    fig, ax = plt.subplots(1, 1)

    # Colors
    colors = [(31, 119, 180), (23, 190, 207), (214, 39, 40)]
    colors = [get_rgb(color) for color in colors]

    for case_type, color in zip(case_types, colors):
        dates = get_dates(data, case_type)
        num_cases = get_num_cases(data, case_type, country, province)
        ax.plot(dates, num_cases, color=color)

        # No legend
        ax.text(dates[-1], num_cases[-1],
                '  {} ({:,})'.format(case_type.capitalize(),
                                   num_cases[-1]),
                color=color, ha='left', va='center')

        # x axis
        ax.set_xlabel('End of month')
        ax.set_xticks(get_end_months(dates))
        ax.set_xticklabels([format_datetime(end_month)
                            for end_month in get_end_months(dates)])
        ax.xaxis.set_tick_params(direction='in')

        # y axis
        ax.set_ylabel('Number of cases')
        ax.yaxis.set_tick_params(direction='in')
        ax.set_yscale('log')

    # Set graph title
    ax.set_title(get_title(country, province))

    sns.despine(ax=ax)

    fig.tight_layout()
    path = 'plots/case_by_country.pdf'
    fig.savefig(path, bbox_inches='tight')
    print('Saved to {}'.format(path))


def plot_active_cases(data, country, province):
    """
    plots active cases (confirmed less recovered and death)
    """
    # Create a data frame with number of active cases
    active = (data['confirmed'].iloc[:, 4:] -
              data['recovered'].iloc[:, 4:] -
              data['death'].iloc[:, 4:])

    # Copy the identifying columns on geography
    identifier = data['confirmed'][['province/state',
                                    'country/region', 'lat', 'long']]

    # Append two dataframes
    active = pd.concat([identifier, active], axis=1)

    # Append active cases into master data
    data['active'] = active

    # Plot active cases by country
    fig, ax = plt.subplots(1, 1)
    case_type = 'active'

    # Choose color scheme
    color_active = get_rgb((188, 189, 34))

    dates = get_dates(data, case_type)
    num_cases = get_num_cases(data, case_type, country, province)
    ax.plot(dates, num_cases, color=color_active)

    ax.text(dates[-1], num_cases[-1], '{:,.0f}'.format(num_cases[-1]),
            color = color_active, ha='left', va='center')

    # x axis
    ax.set_xlabel('End of month')
    ax.set_xticks(get_end_months(dates))
    ax.set_xticklabels([format_datetime(end_month)
                        for end_month in get_end_months(dates)])
    ax.xaxis.set_tick_params(direction='in')

    # y axis
    ax.set_ylabel('Number of active cases')
    ax.yaxis.set_tick_params(direction='in')
    ax.set_yscale('log')

    # Set graph title
    ax.set_title(get_title(country, province))

    sns.despine(ax=ax)

    fig.tight_layout()
    path = 'plots/active_case_by_country.pdf'
    fig.savefig(path, bbox_inches='tight')
    print('Saved to {}'.format(path))


def plot_new_cases(data, country, province):
    """
    plot daily new cases
    """
    # Copy the identifying columns on geography
    identifier = data['confirmed'][[
                      'province/state', 'country/region', 'lat', 'long']]

    # Insert first column
    col = data['confirmed'].iloc[:, 4]
    daily_new = col.to_frame()

    for i in range(5, len(data['confirmed'].columns)):
        col = pd.Series(data['confirmed'].iloc[:, i] -
                        data['confirmed'].iloc[:, i-1])
        daily_new[data['confirmed'].columns[i]] = col

    # Append with geography identifier
    daily_new = pd.concat([identifier, daily_new], axis=1)

    # Append active cases into master data
    data['daily_new'] = daily_new

    # Plot active cases by country

    fig, ax = plt.subplots(1, 1)
    case_type = 'daily_new'

    # Choose color scheme
    color_daily = get_rgb((44, 160, 44))

    dates = get_dates(data, case_type)
    num_cases = get_num_cases(data, case_type, country, province)
    ax.bar(dates, num_cases, color=color_daily)

    # x axis
    ax.set_xlabel('End of month')
    ax.set_xticks(get_end_months(dates))
    ax.set_xticklabels([format_datetime(end_month)
                        for end_month in get_end_months(dates)])
    ax.xaxis.set_tick_params(direction='in')

    # y axis
    ax.set_ylabel('Number of new cases')
    ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])
    ax.yaxis.set_tick_params(direction='in')
    
    # Set graph title
    ax.set_title(get_title(country, province))

    sns.despine(ax=ax)

    fig.tight_layout()
    path = 'plots/daily_case_by_country.pdf'
    fig.savefig(path, bbox_inches='tight')
    print('Saved to {}'.format(path))


def get_first(data, n, case_type, country, province=None):
    """
    returns the time-series data frame with no. of confirmed cases from the
    day of nth infection or no. of deaths since nth death
    """
    if province is None:
        # Extract the data frame with the country of interest
        result = (data[case_type][data[case_type]['country/region'] == country].iloc[:, 4:]).sum(axis=0)

    else:
        condition = ((data[case_type]['country/region'] == country) &
                     (data[case_type]['province/state'] == province))
        result = data[case_type][condition].iloc[:, 4:]
        result = result.iloc[0, :]

    if sum(result < n) == len(result):
        result = None
    else:
        # Extract the dates with at least n cases
        result = result.loc[result >= n]

        # Create a series of number of days since nth infection
        num_days = pd.Series(range(0, len(result)))
        num_days.index = result.index

        # Combine series of at least n cases and no. of days since nth infection/ death
        result = pd.concat([num_days, result], axis=1)
        result.columns = ['num_days', 'cases']

    return result


def plot_first(data, n, case_type, country, province):
    """
    returns the graph of nth infection/ death for a single country/ province
    """

    fig, ax = plt.subplots(1, 1)

    table = get_first(data, n, case_type, country, None)

    if table is None:
        pass
    else:
        # Color
        color = get_rgb((31, 119, 180))

        # Plot the graph
        ax.plot(table['num_days'].values, table['cases'].values,
                color=color)

    # x axis
    ax.set_xlabel('Days since ' + str(n) + 'th ' + case_type + ' case')
    ax.xaxis.set_tick_params(direction='in')

    # y axis
    ax.set_ylabel('Number of cases')
    ax.yaxis.set_tick_params(direction='in')
    ax.set_yscale('log')

    # Set graph title
    ax.set_title(get_title(country, province))

    sns.despine(ax=ax)

    fig.tight_layout()
    path = 'plots/case_first_{}.pdf'.format(case_type)
    fig.savefig(path, bbox_inches='tight')
    print('Saved to {}'.format(path))


def plot_compare_first(data, n, case_type, countries, path=None):
    """
    returns the graph after nth infection or death for a few countries
    """

    fig, ax = plt.subplots(1, 1)

    palette10 = [(31, 119, 180), (255, 127, 14),
                 (44, 160, 44),  (214, 39, 40),
                 (148, 103, 189),  (140, 86, 75),
                 (227, 119, 194),  (127, 127, 127),
                 (188, 189, 34),  (23, 190, 207)]

    for country in countries:
        table = get_first(data, n, case_type, country, None)
        if table is None:
            pass
        else:
            # Get color
            color = get_rgb(palette10[countries.index(country)])
            ax.plot(table['num_days'].values,
                    table['cases'].values,
                    color=color)

            # No legend
            ax.text(table['num_days'].values[-1],
                    table['cases'].values[-1],
                    '  ' + country,
                    ha='left', va='center', color=color)

    # x axis
    ax.set_xlabel('Days since ' + str(n) + 'th ' + case_type + ' case')
    ax.xaxis.set_tick_params(direction='in')

    # y axis
    ax.set_ylabel('Number of ' + case_type + ' cases')
    ax.yaxis.set_tick_params(direction='in')
    ax.set_yticklabels(['{:,}'.format(int(x)) 
                        for x in ax.get_yticks().tolist()])
    ax.set_yscale('log')

    # title
    ax.set_title('As of ' + datetime.today().strftime('%Y-%m-%d'))

    sns.despine(ax=ax)

    fig.tight_layout()

    if path is None:
        path = 'plots/compare_first_{}.pdf'.format(case_type)

    fig.savefig(path, bbox_inches='tight')
    print('Saved to {}'.format(path))


# this means that the code inside is only run when you run
# python analysis.py
# it is NOT run if you do import analysis
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--country', default='US', help=' ')
    parser.add_argument('--province', default=None, help=' ')
    args = parser.parse_args()

    data = get_data()

    # Create plots/ folder
    Path('plots/').mkdir(exist_ok=True)

    plot_case_by_country(data, args.country, args.province)
    plot_active_cases(data, args.country, args.province)
    plot_new_cases(data, args.country, args.province)
    plot_first(data, 100, 'confirmed', args.country, args.province)
    countries = ['US', 'United Kingdom', 'Singapore',
                 'China', 'Italy', 'Korea, South',
                 'Germany', 'Iran', 'Vietnam', 'Slovakia']
    plot_compare_first(data, 100, 'confirmed', countries)
    plot_compare_first(data, 25, 'death', countries)
    plot_compare_first(data, 25, 'death', countries,
                       'corona_deaths.png')
