# University admission prediction

This repository contains a basic implementation of a secure API for predicting a student's chance of admission to a university

This is how the project is built:

```bash       
├── examen_bentoml             
│   ├── data                <- data used for the project
│   │   ├── processed       <- model-ready data
│   │   └── raw             <- raw data
│   ├── models              <- models or artifacts used in the model
│   ├── notebooks           <- exploratory data analysis files
│   ├── src                 <- directory containing all scripts needed to build the model and API
│   │   ├── data            <- downloading and cleaning data 
│   │   ├── models          <- model training
│   │   ├── predit          <- model inference
│   │   └── test            <- testing model inference
│   ├── bentofile.yaml      <- bentoML configuration file
│   ├── requirements.txt    <- dependencies required for the project
│   ├── setup.sh            <- automatic project start-up
│   └── README.md           
```

## How to Run the Project

1. Install the required dependencies:
```bash
pip install -r requirements.txt

```

2. Run the setup script:
```bash
./setup.sh

```

If you encounter permission issues, make the script executable first:
```bash
chmod 700 setup.sh

```

3. Follow the prompts to select your preferred server option:
- Enter 1 to use the BentoML development server.
- Enter 2 to use the Docker container (ensure Docker is installed on your machine).