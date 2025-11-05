import pandas as pd

#example usage
# python3 src/clean_dataset.py

dataset = pd.read_csv('./data/DB for chromophore_Sci_Data_rev02.csv')
clean_dataset = dataset.iloc[:, [0, 1, 2, 3, 4]].dropna()
# drop rows with Solvent="gas"
clean_dataset = clean_dataset[clean_dataset['Solvent'] != 'gas']

clean_dataset.to_csv('./data/clean_chromophore_data.csv', index=False)