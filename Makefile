COMPOSE ?= docker compose

.PHONY: up down logs test verify help

help:
	@echo "make up      build and start the whole stack (frontend :3000, backend :8000)"
	@echo "make down    stop it (the SQLite volume is kept; use 'make clean' to delete it)"
	@echo "make logs    follow container logs"
	@echo "make test    run backend and frontend tests (needs Python and Node locally)"
	@echo "make verify  check a running backend against the API contract"

.env:
	@cp .env.example .env
	@echo ">> Created .env from .env.example with PLACEHOLDER keys. Edit LLM_API_KEYS in .env, then run 'make up' again to apply."

up: .env
	$(COMPOSE) up --build -d
	@echo ""
	@echo "Frontend: http://localhost:3000"
	@echo "Backend:  http://localhost:8000/health"

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v

logs:
	$(COMPOSE) logs -f

test:
	cd backend && pip install -q -r requirements-dev.txt && python -m pytest -q
	cd frontend && npm ci --no-audit --no-fund && npm test

verify:
	python3 scripts/verify_contract.py $${BACKEND_URL:-http://localhost:8000} --origin $${FRONTEND_URL:-http://localhost:3000}
