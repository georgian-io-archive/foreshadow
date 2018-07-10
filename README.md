# foreshadow
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/georgianpartners/foreshadow/blob/master/LICENSE "License")
[![BuildStatus](https://travis-ci.org/georgianpartners/foreshadow.svg?branch=master "Build Status")](https://travis-ci.org/georgianpartners/foreshadow)
[![Coverage](https://coveralls.io/repos/github/georgianpartners/foreshadow/badge.svg?branch=development "Coverage")](https://coveralls.io/github/georgianpartners/foreshadow)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# Manually running tests
* setup pyenv for 3.5.5 
* setup pyenv for 3.6.5
* setup a test pyenv virtualenv
  * install test_requirements to it
```bash
(testenvforeshadow) $ pip install -r test_requirements.txt
(testenvforeshadow) $ tox -r
```