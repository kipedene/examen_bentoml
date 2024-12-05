#!/bin/bash
python3 src/data/import_raw_data.py
python3 src/data/prepare_data.py data/processed
python3 src/models/train_model.py
bentoml build
bentoml containerize lr_pred_service:latest --docker-image-tag lr_pred_service:latest

while true; do
    echo "What do you want to do?"
    echo "1) Launch the BentoML server"
    echo "2) Launch the Docker container"
    read -p "Enter your choice (1 or 2): " choice

    if [ "$choice" -eq 1 ]; then
        echo "Launching the BentoML server..."
        bentoml serve src.predict.service:lr_service --reload
        break  
    elif [ "$choice" -eq 2 ]; then
        echo "Launching the Docker container..."
        docker run -p 3000:3000 lr_pred_service:latest
        break  
    else
        echo "Invalid choice. Please enter 1 or 2."
    fi
done