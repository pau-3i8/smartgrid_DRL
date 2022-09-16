import numpy as np, pandas as pd

def data_aggregation(df, data_list):
    """
    Column aggregation
    """
    # temporal dataframe
    temp_df = df.copy()
    
    new_df = pd.DataFrame()
    new_df.index = temp_df.index
    for aggregation in data_list:
        if aggregation in temp_df.columns.get_level_values(0):
            new_df[aggregation] = temp_df.loc[:, aggregation]
            df = df.drop(columns = aggregation) #borra la columna
        else: print(aggregation, 'not in DataFrame')
    # skipna=False, otherwise NaNs will be zeros after summation
    # skipna=True if data is properly preprocessed without nans
    sum_col = new_df.sum(axis='columns', skipna=True).to_frame().round(0); del temp_df
    
    return sum_col
    
def preprocess():

    power_df = pd.read_csv('original_datasets/energy_dataset.csv')
    weather_df = pd.read_csv('original_datasets/weather_60min.csv')
    
    # Fill missing data
    power_df = power_df.interpolate(method='polynomial', order=5)
    weather_df = weather_df.interpolate(method='polynomial', order=5)
    
    aggregation_list = ['generation solar', 'generation biomass', 'generation fossil brown coal/lignite', 'generation fossil coal-derived gas', 'generation fossil gas', 'generation fossil hard coal', 'generation fossil oil', 'generation fossil oil shale', 'generation fossil peat', 'generation geothermal', 'generation hydro pumped storage aggregated', 'generation hydro pumped storage consumption', 'generation hydro run-of-river and poundage', 'generation hydro water reservoir', 'generation marine', 'generation nuclear', 'generation other', 'generation other renewable', 'generation waste', 'generation wind offshore', 'generation wind onshore']    
    generation_total = data_aggregation(power_df, aggregation_list) #without solar
    
    #set_index
    df = pd.DataFrame()
    df.index = power_df.index #copy index
    
    df['gen_total'] = generation_total
    df['gen_solar'] = power_df.loc[:, 'generation solar']
    df['demand'] = power_df.loc[:, 'total load actual']
    df['demand_forecast'] = power_df.loc[:, 'total load forecast']
    df['ir'] = weather_df.loc[:, 'irradiance_surface']
    # indexation definition always at the end
    df.index = pd.to_datetime(power_df.iloc[:, 0])
    df.index.name = 'Timestamp' #index name
    
    print('Nans percentage \n', df.isna().sum()/len(df)*100)
    
    df.to_csv('dataset.csv')
    
if __name__ == "__main__":
    preprocess()
