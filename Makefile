CWD=$(shell pwd)
PKG=foreshadow
TST=tests

lint:
	pre-commit run --all-files

clean:
	find ./$(PKG) -name "*.pyc" -exec rm -rfv {} \;

test:
	pytest --pdb

coverage:
	coverage html; open htmlcov/index.html

.PHONY: clean lint test coverage