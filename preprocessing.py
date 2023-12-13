import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

def main():
    df = pd.read_csv('US_Accidents_March23.csv')
    print("read")
    
    # delete pre-selected features
    df = df.drop(
        columns=['ID',
                 'End_Lat', 
                 'End_Lng',
                 'Description',
                 'Street',
                 'City',
                 'County',
                 'Zipcode',
                 'Country',
                 'Airport_Code',
                 'Weather_Timestamp',
                 'Weather_Condition',
                 'Wind_Direction',
                 'Timezone',
                 'Turning_Loop'
                 ])
    
    # replace missing values with 0
    df['Precipitation(in)'] = df['Precipitation(in)'].fillna(0)
    # replace missing with mean 
    df['Temperature(F)'] = df['Temperature(F)'].fillna(df['Temperature(F)'].mean())
    df['Wind_Chill(F)'] = df['Wind_Chill(F)'].fillna(df['Wind_Chill(F)'].mean())
    df['Humidity(%)'] = df['Humidity(%)'].fillna(df['Humidity(%)'].mean())
    df['Pressure(in)'] = df['Pressure(in)'].fillna(df['Pressure(in)'].mean())
    df['Visibility(mi)'] = df['Visibility(mi)'].fillna(df['Visibility(mi)'].mean())
    df['Wind_Speed(mph)'] = df['Wind_Speed(mph)'].fillna(df['Wind_Speed(mph)'].mean())
    # drop rows with missing values
    df = df.dropna(subset=['Sunrise_Sunset','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight'])

    # one hot encoding
    # state
    one_hot = pd.get_dummies(df['State'])
    df = df.join(one_hot)
    # source
    one_hot = pd.get_dummies(df['Source'])
    df = df.join(one_hot)
    # drop encoded columns
    df = df.drop(columns=[
        'State',
        'Source'
    ])

    # convert true/false to 1/0
    df = df.replace(to_replace="True", value=1)
    df = df.replace(to_replace="False", value=0)
    # convert day/night to 1/0
    df = df.replace(to_replace="Day", value=1)
    df = df.replace(to_replace="Night", value=0)

    # extract year, month, hour and duration from time
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['End_Time'] = pd.to_datetime(df['End_Time'])
    df['Year'] = df['Start_Time'].dt.year
    df['Month'] = df['Start_Time'].dt.month
    df['Hour'] = df['Start_Time'].dt.hour
    df['Duration'] = (df['End_Time'] - df['Start_Time'])/np.timedelta64(1, 'm')
    # drop start and end time
    df = df.drop(columns=[
        'Start_Time',
        'End_Time'
    ])

    # extract target 
    target = df['Severity']
    df = df.drop(columns=['Severity'])

    # standard scale features
    print('scaling')
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    
    # last check for nulls
    print(df.columns[df.isnull().any()].tolist())
    
    # convert to csv
    print('to csv')
    target.to_csv('y.csv', index=False)
    df.to_csv('data.csv', index=False)

if __name__ == "__main__":
    main()