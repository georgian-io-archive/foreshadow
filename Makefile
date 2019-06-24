CWD=$(shell pwd)
PKG=foreshadow

clean:
	find ./$(PKG) -name "*.pyc" -exec rm -rfv {} \;

test:
	tox -r

coverage:
	coverage html; open htmlcov/index.html

.PHONY: test clean coverage