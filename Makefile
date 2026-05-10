up:
	docker compose up --build -d

up-gpu:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d

ps:
	docker compose ps

shell:
	docker compose exec slm-dev bash

dataset:
	docker compose exec slm-dev python scripts/prepare_dataset.py

check:
	docker compose exec slm-dev python scripts/check_env.py

train:
	docker compose exec slm-dev python scripts/train_lora.py

merge:
	docker compose exec slm-dev python scripts/merge_lora.py

infer:
	docker compose exec slm-dev python scripts/infer.py --prompt "Create a Java 21 DTO named OrderRequest with Lombok Builder and Jakarta Validation."

api-restart:
	docker compose restart slm-api

api-health:
	curl -s http://localhost:8000/health | python -m json.tool

api-test:
	curl -s http://localhost:8000/generate \
		-H "Content-Type: application/json" \
		-d '{"prompt":"Create a Java DTO named PatientIntakeRequest with Lombok Builder and Jakarta Validation.","max_new_tokens":220}' | python -m json.tool


ollama-create:
	docker compose run --rm -e OLLAMA_HOST=http://ollama:11434 ollama-setup

ollama-recreate:
	docker compose exec ollama ollama rm java-dto-assistant || true
	docker compose run --rm -e OLLAMA_HOST=http://ollama:11434 ollama-setup

ollama-ready:
	docker compose exec -e OLLAMA_BASE_URL=http://ollama:11434 slm-dev python scripts/wait_for_ollama.py

ollama-tags:
	curl -s http://localhost:11434/api/tags | python -m json.tool

ollama-build-custom:
	python scripts/build_custom_ollama_model.py

ollama-build-custom-docker:
	docker compose exec -e OLLAMA_BASE_URL=http://ollama:11434 -e OLLAMA_MODEL=java-dto-assistant slm-dev \
		python scripts/build_custom_ollama_model.py


ollama-test:
	docker compose exec -e OLLAMA_BASE_URL=http://ollama:11434 -e OLLAMA_MODEL=java-dto-assistant slm-dev \
		python scripts/query_ollama.py --prompt "Create a Java 21 DTO named OrderRequest with Lombok Builder, Swagger Schema examples, and Jakarta Validation."

ollama-test-host:
	python scripts/query_ollama.py --prompt "Create a Java 21 DTO named OrderRequest with Lombok Builder, Swagger Schema examples, and Jakarta Validation."

ollama-curl:
	curl -s http://localhost:11434/api/generate \
		-H "Content-Type: application/json" \
		-d '{"model":"java-dto-assistant","prompt":"Create a Java DTO named EvidenceUploadRequest with Lombok and Jakarta Validation.","stream":false}' | python -m json.tool

logs:
	docker compose logs -f slm-dev slm-api ollama ollama-setup

down:
	docker compose down

clean:
	docker compose down -v
	rm -rf outputs
