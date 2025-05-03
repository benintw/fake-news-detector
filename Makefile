
.PHONY: clean train validate predict

clean:
	@echo "Cleaning all __pycache__ directories and .pyc files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete!"

train:
	@echo "Training model..."
	@uv run python -m scripts.train
	@echo "Training complete!"

validate:
	@echo "Validating model..."
	@uv run python -m scripts.evaluate
	@echo "Validation complete!"

predict:
	@echo "Predicting..."
	@uv run python -m scripts.predict --file "./sample_news.txt"
	@echo "Prediction complete!"
