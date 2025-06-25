CURRENT_DIR = $(shell pwd)


prepare-dirs:
	mkdir -p ${CURRENT_DIR}/data || true && \
	mkdir -p ${CURRENT_DIR}/data/models || true && \
	mkdir -p ${CURRENT_DIR}/data/faiss_index || true && \
	mkdir -p ${CURRENT_DIR}/data/plaintext_index || true && \
	mkdir -p ${CURRENT_DIR}/data/docs || true

run-local:
	python src/main.py --input ${REPO}

# update-index:
#	ROOT_DATA_DIR=${CURRENT_DIR}/data \
#	python src/update_index.py --input ${REPO}

update-index:
	docker run \
	-e PYTHONPATH=/app/src \
	--env-file "${CURRENT_DIR}/.env" \
	-v "${CURRENT_DIR}/src:/app/src" \
	-v "${CURRENT_DIR}/data:/app/data" \
	-v "${REPO}:/app/data/docs" \
	sweedbrain-api:dev "uv" run src/update_index.py --input /app/data/docs

build-api-dev:
	docker build --target development -t sweedbrain-api:dev \
	${CURRENT_DIR}

run-api-dev: prepare-dirs
	docker run -p 8080:8000 \
	-e PYTHONPATH=/app/src \
	--env-file "${CURRENT_DIR}/.env" \
	-v "${CURRENT_DIR}/src:/app/src" \
	-v "${CURRENT_DIR}/data:/app/data" \
	-v "${REPO}:/app/data/docs" \
	sweedbrain-api:dev

run-jupyter:
	DATA_DIR=${CURRENT_DIR}/data \
	PYTHONPATH=${CURRENT_DIR}/src \
	jupyter notebook jupyter_notebooks --ip 0.0.0.0 --port 8889 --NotebookApp.token='' --NotebookApp.password='' --allow-root --no-browser 
