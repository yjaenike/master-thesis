# Data
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Plotting
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# Other
import time

class PreprocessingPipeline():
    '''
    -> Outlier handling
    -> Missing-value handling
    '''
    
    def __init__ (self, fill_type, clip_quantile_value) -> None:
        self.fill_type = fill_type # nan, locf, nocb, ema
        self.clip_quantile_value = clip_quantile_value
        
        # will be set up later on
        self.upper_quantiles_clip = None
        self.lower_quantiles_clip = None
        self.mean = None
        self.median = None
        self.std  = None
        self.powerTransformer = None
        self.metadata = None
        
    def fit(self, df: pd.DataFrame) -> None:
        '''
        Calculate quantiles, mean, meadian and std of the dataframe for the transformation steps
        
            Parameters:
            -----------
                df (pd.DataFrame): the input dataframe we are going to gather our values from.
        '''
        
        # remove some columns and set feature columns
        #candidates = ['pat_id', 'rel_time']
        #df = df.drop([x for x in candidates if x in df.columns], axis=1)
        self.feature_columns = df.columns
        
        # Calculate quantile values
        self.upper_quantiles_clip = df.quantile(self.clip_quantile_value, axis=0).to_numpy()
        self.lower_quantiles_clip = df.quantile(1 - self.clip_quantile_value, axis=0).to_numpy()
        
        # Apply quantiles
        df_clipped = self.__clip_outliers(df)
        
        # calc mean, median and std using the clipped dataframe
        self.mean = np.nanmean(df_clipped._get_numeric_data(), axis=0) 
        self.median = np.nanmedian(df_clipped._get_numeric_data(), axis=0)
        self.std = np.nanstd(df_clipped._get_numeric_data(), axis=0)
        
        # add metadata for dataframe
        self.metadata = pd.DataFrame(self.feature_columns, columns=["column_name"])
        self.metadata = self.metadata.set_index('column_name')
        self.metadata["nan"] = df_clipped.isna().sum()
        self.metadata["mean"] = self.mean
        self.metadata["median"] = self.median
        self.metadata["std"] = self.std
        self.metadata["kurtosis"] = df_clipped.kurtosis()
        self.metadata["skew"] = df_clipped.skew()


        
    
    def transform(self, df):
        '''
        Transformed the dataframe object, by clipping the outliers and filling in the missing values
        
            Parameters:
            -----------
                df (pd.DataFrame): the input dataframe we are going to transform.

            Returns:
            --------
                df_t (pd.DataFrame): the transformed dataframe
        ''' 
        
        df_t = df.loc[:, self.feature_columns]
        
        df_t = self.__clip_outliers(df_t)
        df_t = self.__fill_missing_values(df_t, self.fill_type)
        df_t = self.__feature_scale(df_t)
        df_t = self.__power_transform(df_t)
        
        #df_t.insert(0, 'rel_time', df['rel_time'])
        #df_t.insert(0, 'pat_id', df['pat_id'])
        
        
        return df_t
    
    def fit_transform(self,df):
        '''
        Fit Preprocessor to the data and transforme the dataframe object
        
            Parameters:
            -----------
                df (pd.DataFrame): the input dataframe we are going to fit to and transform.

            Returns:
            --------
                df_t (pd.DataFrame): the transformed dataframe
        '''
        # fit to dataframe
        self.fit(df.copy())
        
        #return the transformed dataframe
        return self.transform(df)
    
    
    # Power Transform
    def __power_transform(self, df):

        #instatiate 
        self.powerTransformer = PowerTransformer(method='yeo-johnson', standardize=True,) 
        
        #Fit the data to the powertransformer
        skl_yeojohnson = self.powerTransformer.fit(df)
        self.metadata["yeojoh_lambdas"] = skl_yeojohnson.lambdas_
        
        #Transform the data 
        skl_yeojohnson = self.powerTransformer.transform(df)
        
        df_t = pd.DataFrame(data=skl_yeojohnson, columns=self.feature_columns)
        
        return df_t
    
    # Feature Scaling / noramlisation
    def __feature_scale(self, df):
        ''' Standardize features by removing the mean and scaling to unit variance.  '''
        
        df_t = (df - df.mean()) / df.std()
        
        return df_t
    
    # Outlier handling
    def __clip_outliers(self, df):
        ''' Clips each column based on the computet quantile values '''        
        
        df_t = df.clip(self.lower_quantiles_clip, self.upper_quantiles_clip)
        
        return df_t
    
    # Missing value handling
    def __fill_missing_values(self, df, fill_type):
        ''' Based of the fill type defined, fill the nan values '''
        
        match fill_type:
            case "nan":
                df_t = self.__fill_nan(df)
            
            case "median":
                df_t = self.__fill_median(df)
                
            case "locf":
                df_t = self.__fill_locf(df)
                
            case "nocb":
                df_t = self.__fill_nocb(df)
                
            case "ema":
                df_t = self.__fill_ema(df, 0.9, df.mean())
            
            case "ema_fast":
                columns = df.columns
                data = df.to_numpy()
                mean_array = df.mean().to_numpy()
                
                df_t = self.__fill_ema_fast(data, 0.9, mean_array)
                df_t = pd.DataFrame(df_t, columns = columns)
                
            case _:
                df_t = self.__fill_nan(df)
        
        return df_t
    
    def __fill_nan(self, df):
        ''' Keep nan values and do no imputation '''  
        return df
    
    def __fill_median(self, df):
        ''' Replace nan with median of that column '''
        
        df_t = df.mask(df.isna(),df.median(), axis = 1)
        
        return df_t
    
    def __fill_locf(self, df):
        ''' Last Observation Carried Forward: the missing value is imputed using the values before it in the time series. '''  
        
        # first forwardfill then backfill
        df_t = df.fillna(method="ffill").fillna(method='bfill')
        
        return df_t 
    
    def __fill_nocb(self, df):
        ''' Next Observation Carried Backward: the missing value is imputed using the values coming next it in the time series. '''  
        
        # first backfill then forwadfill
        df_t = df.fillna(method="bfill").fillna(method='ffill')
        
        return df_t 
    
    def __fill_linear(self, df):
        ''' Linear interpolation beteen values '''
        
        df_t = df.interpolate(method='linear')
        
        return df_t
    
    def __fill_ema(self, df, alpha, mean):
        ''' Exponential moving average:
            X × k + EMA(y) × (1−k) 

            X = current value
            k = alpha
            EMA(y) = row of yesterday  
        '''

        # get the first row of the dataframe and fill with mean to initialize it
        ema_yesterday = mean #df.iloc[:1]

        # create an array the same shape as the patition this will be used to fill with calculations
        ema_df = pd.DataFrame().reindex_like(df)

        # for each row in the array, Set the nan values to 0 
        for index, row in df.fillna(0).iterrows():
            
            # Ema Formula:
            ema_today = row * alpha + (1 - alpha) * ema_yesterday

            # copy the row into the complete ema dataframe.
            ema_df.iloc[index,:] = ema_today
            
            # repalce yeserdays row with current row for next step
            ema_yesterday = ema_today.copy()

        df_t = df.mask(df.isna(), ema_df)

        return df_t
    
    def __fill_ema_fast(self,data: np.ndarray, alpha: float,mean_array: np.ndarray):
        ''' Exponential moving average:
            X × k + EMA(y) × (1−k) 

            X = current value
            k = alpha
            EMA(y) = row of yesterday  
        '''
        # init ema
        ema = np.ones_like(data[0]) * mean_array

        # run ema
        ema_steps = np.ones_like(data)
        for i, data_row in enumerate(data):
            data_row[np.isnan(data_row)] = 0
            ema = alpha * ema + (1 - alpha) * data_row
            ema_steps[i] = ema.copy()
        
        return ema_steps
        