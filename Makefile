CWD=$(shell pwd)
PKG=foreshadow
TST=tests

clean:
	find ./$(PKG) -name "*.pyc" -exec rm -rfv {} \;

test:
	poetry run tox -r

coverage:
	coverage html; open htmlcov/index.html

.PHONY: test clean coverage