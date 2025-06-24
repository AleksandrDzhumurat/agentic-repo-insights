CURRENT_DIR = $(shell pwd)


prepare-dirs:
	mkdir -p ${CURRENT_DIR}/data || true && \
	mkdir -p ${CURRENT_DIR}/data/models || true && \
	mkdir -p ${CURRENT_DIR}/data/docs || true

run-local:
	python src/main.py --input ${REPO}

build-sweedbrain-dev:
	docker build --target development -t sweedbrain-api:dev \
	${CURRENT_DIR}

run-sweedbrain-dev: prepare-dirs
	docker run -p 8080:8000 \
	-e PYTHONPATH=/app/src \
	--env-file "${CURRENT_DIR}/.env" \
	-v "${CURRENT_DIR}/src:/app/src" \
	-v "${CURRENT_DIR}/data:/app/data" \
	-v "${REPO}:/app/data/docs" \
	sweedbrain-api:dev