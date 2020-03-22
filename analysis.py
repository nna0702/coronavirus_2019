import urllib.request
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('default')
plt.rcParams['font.sans-serif'] = "Arial"


def get_data():
    # Name URLs
    confirmed_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
    recovered_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv'
    death_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv'

    # Create a dictionary of file name and url name
    file_names = ['confirmed.csv', 'recovered.csv', 'death.csv']
    urls = [confirmed_url, recovered_url, death_url]
    case_types = ['confirmed', 'recovered', 'death']
    data = {}

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
    case_types = ['confirmed', 'recovered', 'death']
    # Plot cases by country
    fig, ax = plt.subplots(1, 1)

    # Colors
    colors = [(31, 119, 180), (23, 190, 207), (214, 39, 40)]
    colors = [get_rgb(color) for color in colors]

    for case_type, color in zip(case_types, colors):
        dates = get_dates(data, case_type)[:-1]
        num_cases = get_num_cases(data, case_type, country, province)[:-1]
        ax.plot(dates, num_cases, color=color)

        # No legend
        ax.text(dates[-1],
                num_cases[-1],
                case_type,
                color=color)

        # x axis
        ax.set_xlabel('End of month')
        ax.set_xticks(get_end_months(dates))
        ax.set_xticklabels([format_datetime(end_month)
                            for end_month in get_end_months(dates)])
        ax.xaxis.set_tick_params(direction='in')

        # y axis
        ax.set_ylabel('Number of cases')
        ax.yaxis.set_tick_params(direction='in')

    # Set graph title
    ax.set_title(get_title(country, province))

    sns.despine(ax=ax)

    fig.tight_layout()
    path = 'case_by_country.pdf'
    fig.savefig(path, bbox_inches='tight')
    print('Saved to {}'.format(path))


def plot_active_cases(data, country, province):
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

    # x axis
    ax.set_xlabel('End of month')
    ax.set_xticks(get_end_months(dates))
    ax.set_xticklabels([format_datetime(end_month)
                        for end_month in get_end_months(dates)])
    ax.xaxis.set_tick_params(direction='in')

    # y axis
    ax.set_ylabel('Number of active cases')
    ax.yaxis.set_tick_params(direction='in')

    # Set graph title
    ax.set_title(get_title(country, province))

    sns.despine(ax=ax)

    fig.tight_layout()
    path = 'active_case_by_country.pdf'
    fig.savefig(path, bbox_inches='tight')
    print('Saved to {}'.format(path))


def plot_new_cases(data, country, province):
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
    ax.plot(dates, num_cases, color=color_daily)

    # x axis
    ax.set_xlabel('End of month')
    ax.set_xticks(get_end_months(dates))
    ax.set_xticklabels([format_datetime(end_month)
                        for end_month in get_end_months(dates)])
    ax.xaxis.set_tick_params(direction='in')

    # y axis
    ax.set_ylabel('Number of new cases')
    ax.yaxis.set_tick_params(direction='in')

    # Set graph title
    ax.set_title(get_title(country, province))

    sns.despine(ax=ax)

    fig.tight_layout()
    path = 'daily_case_by_country.pdf'
    fig.savefig(path, bbox_inches='tight')
    print('Saved to {}'.format(path))


# this means that the code inside is only run when you run
# python analysis.py
# it is NOT run if you do import analysis
if __name__ == '__main__':
    data = get_data()
    plot_case_by_country(data, 'Italy', None)
    plot_active_cases(data, 'US', 'California')
    plot_new_cases(data, 'Vietnam', None)