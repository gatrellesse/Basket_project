# src/Makefile

.PHONY: setup run_pipeline

# Step to make the script executable
setup:
	chmod +x run_pipeline.sh

# Step to run the pipeline script
run_pipeline: setup
	./run_pipeline.sh
run:
	./run_pipeline.sh
