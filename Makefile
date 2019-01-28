SHELL := /bin/bash
.PHONY: install clean test lint

lint:
	@echo "Lint not implemented yet..."

clean:
	rm -rf *.egg-info
	rm -rf .eggs
	rm -rf advi/__pycache__
	rm -rf advi/transformations/__pycache__
	rm -rf tests/__pycache__
	rm -rf dist

install: test
	python3 -m pip install . --user

test:
	python3 setup.py test
