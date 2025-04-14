run_server_setup:
	@DATA_UPLOAD_MAX_NUMBER_FILES=15000 nohup label-studio --no-browser --port 1234 > label_studio.out 2>&1 &
	@nohup jupyter lab  > jupyter_out.out 2>&1 &
	@nohup mlflow server --backend-store-uri runs/mlflow  > mlflow.out 2>&1 &
	@cd label-studio-ml-backend/label_studio_ml/examples/yolo && \
	sudo docker compose up -d --build
