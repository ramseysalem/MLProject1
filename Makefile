# Makefile in the root directory

.PHONY: start backend frontend

start: setup backend frontend

setup:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt

backend:
	@echo "Starting Flask backend..."
	@cd api && python app.py &

frontend:
	@echo "Starting React frontend..."
	@cd warehouse-clustering-ui && npm start