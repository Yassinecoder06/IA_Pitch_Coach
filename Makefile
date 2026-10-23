.PHONY: dev api web test build docker-up docker-down clean

dev:
	@echo "Starting development backend and web server..."
	(trap 'kill 0' SIGINT; uvicorn backend.main:app --reload --port 8000 & cd web && npm run dev)

api:
	uvicorn backend.main:app --reload --port 8000

web:
	cd web && npm run dev

test:
	pytest tests/ -v

build:
	cd web && npm run build

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf data/*.db __pycache__ backend/**/__pycache__ web/dist
