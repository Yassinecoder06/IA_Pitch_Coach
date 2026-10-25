.PHONY: dev api web test test-live test-web e2e lint build migrate docker-up docker-down clean

dev:
	@echo "Starting development backend and web server..."
	(trap 'kill 0' SIGINT; uvicorn backend.main:app --reload --port 8000 & cd web && npm run dev)

api:
	uvicorn backend.main:app --reload --port 8000

web:
	cd web && npm run dev

test:
	pytest tests/ -m "not live"

test-live:  # needs GEMINI_API_KEY and/or OLLAMA_API_KEY
	pytest tests/ -m live -v

test-web:
	cd web && npm run typecheck && npm run lint && npm run test

e2e: build  # real backend + Chrome; set GEMINI_API_KEY / OLLAMA_API_KEY for the live-AI checks
	cd web && npx playwright test

lint:
	ruff check backend tests
	cd web && npm run lint

migrate:
	python -m backend.cli migrate

build:
	cd web && npm run build

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf data/*.db __pycache__ backend/**/__pycache__ web/dist
