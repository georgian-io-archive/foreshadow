foreshadow
==========

|License| |BuildStatus| |Coverage| |Code style: black|

Manually running tests
======================

-  setup pyenv for 3.5.5
-  setup pyenv for 3.6.5
-  setup a test pyenv virtualenv

   -  install test_requirements to it

   .. code:: bash

      (testenvforeshadow) $ pip install -r test_requirements.txt
      (testenvforeshadow) $ tox -r

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://github.com/georgianpartners/foreshadow/blob/master/LICENSE
.. |BuildStatus| image:: https://travis-ci.org/georgianpartners/foreshadow.svg?branch=master
   :target: https://travis-ci.org/georgianpartners/foreshadow
.. |Coverage| image:: https://coveralls.io/repos/github/georgianpartners/foreshadow/badge.svg?branch=development
   :target: https://coveralls.io/github/georgianpartners/foreshadow
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black