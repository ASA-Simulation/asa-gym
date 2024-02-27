.PHONY: clean configure build install proto help

.DEFAULT_GOAL := help


clean: ## Clean all generated build files in the project.
	rm -rf asagym.egg-info/


configure: ## Configure the project for building.
	python3 -m venv ../dist/venv


build: ## Build all targets in the project.
	bash -c "source ../dist/venv/bin/activate && \
		pip install grpcio-tools && \
		python -m grpc_tools.protoc -I=asagym/proto \
			--python_out=asagym/proto \
			--pyi_out=asagym/proto \
			asagym/proto/simulator.proto"


install: ## Install all targets in the project.
	bash -c "source ../dist/venv/bin/activate && pip install -e ."


help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
