import requests
import os
import logging
from check_structure import check_existing_file, check_existing_folder


def import_raw_data(raw_data_relative_path, filenames, bucket_folder_url):
    """import filenames from bucket_folder_url in raw_data_relative_path"""
    if check_existing_folder(raw_data_relative_path):
        os.makedirs(raw_data_relative_path)
    # download all the files
    for filename in filenames:
        input_file = os.path.join(bucket_folder_url, filename)
        output_file = os.path.join(raw_data_relative_path, filename)
        if check_existing_file(output_file):
            object_url = input_file
            print(f"downloading {input_file} as {os.path.basename(output_file)}")
            try:
                response = requests.get(object_url)
                if response.status_code == 200:
                    # Process the response content as needed
                    content = (
                        response.content
                    )  # Utilisez response.content pour les fichiers binaires
                    with open(output_file, "wb") as file:
                        file.write(content)
                else:
                    print(f"Error accessing the object {input_file}:", response.status_code)
            except Exception as e:
                print(f"An error occurred: {str(e)}")


def main(
    raw_data_relative_path="data/raw",
    filenames=["admission.csv"],
    bucket_folder_url="https://assets-datascientest.s3.eu-west-1.amazonaws.com/MLOPS/bentoml/",
):
    """Upload data from AWS s3 in data/raw"""
    import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)
    logger = logging.getLogger(__name__)
    logger.info("making raw data set")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
