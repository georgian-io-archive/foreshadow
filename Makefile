CWD=$(shell pwd)
PKG=foreshadow
TST=tests

clean:
	find ./$(PKG) -name "*.pyc" -exec rm -rfv {} \;

test:
	poetry run tox -r

.PHONY: test clean