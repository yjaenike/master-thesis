# Data

### Datenbank_Werte.csv
    This is the export from the UKE, MIMIC II and eICU. 
    The data is in a long column format.

### pat_id_data.csv
    This are extra variables for the Datenbank_Werte.csv database. It contains personal data of the patients 
    like gender, hightm age and more

### Datenbank_werte_df.parquet
    This is the export from the UKE, MIMIC II and eICU. 
    The data is in a long column format.
    
### database_values_wide.parquet
    This is the wide version of the UKE database.
    I have pivoted the table, reamoved everything but the UKE data and cleaned the column names
    
### uke_clean.parquet
    This is the left merge of 'database_values_wide' and 'pat_id_data'.
    Additionally some columns have been removed ['database', 'diagnose', 'outcome', 'outcome_house']
    The sex has been Label encoded for all data.