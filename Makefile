SHELL := /bin/bash
.PHONY: install clean test lint

lint:
	@echo "Lint not implemented yet..."

test:
	python3 setup.py test

clean:
	rm -rf *.egg-info
	rm -rf .eggs
	rm -rf advi/__pycache__

install:
	python3 -m pip install . --user
