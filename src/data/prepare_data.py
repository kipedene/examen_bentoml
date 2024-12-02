# -*- coding: utf-8 -*-
import click, logging, shutil, pickle, os
import pandas as pd
from check_structure import check_existing_folder, check_existing_file
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataImporter:
    def __init__(self, filepath='data/raw'):
        self.filepath = filepath

    def load_data(self):
        data = pd.read_csv(f'{self.filepath}/admission.csv')
        data = data.drop('Serial No.', axis=1)
        data = data.rename(
            columns=
                    {
                        "GRE Score":"gre_score",
                        "TOEFL Score":"toefl_score",
                        "University Rating":"university_rating",
                        "SOP":"sop",
                        "LOR ":"lor",
                        "CGPA":"cgpa",
                        "Research":"research",
                        "Chance of Admit ":"chance_of_admit"
                    }
        )

        return data

    def split_train_test(self, df, test_size = .2):
        X = df.drop("chance_of_admit", axis=1)
        y = df["chance_of_admit"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def normalize_data(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_folderpath', type=click.Path())
def main( output_folderpath):

    data_importer = DataImporter()
    df = data_importer.load_data()
    X_train, X_test, y_train, y_test = data_importer.split_train_test(df)
    X_train_scaled, X_test_scaled, = data_importer.normalize_data(X_train, X_test)

    if check_existing_folder(output_folderpath):
        os.makedirs(output_folderpath)
    
    for file, filename in zip([X_train_scaled, X_test_scaled, y_train, y_test], ['X_train_scaled', 'X_test_scaled', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            df = pd.DataFrame(file)
            df.to_csv(output_filepath, index=False)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

