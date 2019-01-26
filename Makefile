SHELL := /bin/bash
.PHONY: install clean test lint

INIT_FILES = advi/__init__.py advi/transformations/__init__.py
SRC_FILES = advi/Model.py advi/transformations/transformations.py	

lint:
	@echo "Lint not implemented yet..."

clean:
	rm -rf *.egg-info
	rm -rf .eggs
	rm -rf advi/__pycache__
	rm -rf advi/transformations/__pycache__
	rm -rf tests/__pycache__

install: test
	python3 -m pip install . --user

test: $(SRC_FILES) $(INIT_FILES)
	python3 setup.py test
