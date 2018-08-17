.. _contrib:

Contributing
============

Project Setup
-------------
Foreshadow has two main branches. Development and master. Each pypi version and their associated commit will be tagged in GitHub. Before each release master will be merged into development and development will be merged back into master. This project follows the `semantic versioning`_ standard.

.. _semantic versioning: https://semver.org/

Issues
------
Please feel free to submit issues to github with any bugs you encounter, ideas you may have, or questions about useage. We only ask that you tag them appropriately as *bug fix*, *feature request*, *usage*. Please also follow the following format::

   #### Description
   Please add a narrative description of your issue.
   
   #### Steps/Code to Reproduce
   Please add the minimum code required to reproduce the issue if possible. If the code is too long, feel free to put it in a public gist and link it in the issue: https://gist.github.com

   #### Expected Results
   Please add the results that you would expect here
   
   #### Actual Results
   Please place the full traceback here
   
   #### Versions
   
   Please run the following snippet in your code and paste the results here.
   
   import platform; print(platform.platform())
   import sys; print("Python", sys.version)
   import numpy; print("NumPy", numpy.__version__)
   import sklearn; print("Scikit-Learn", sklearn.__version__)
   import pandas; print("Pandas", pandas.__version__)
   import foreshadow; print("Foreshadow", foreshadow.__version__)

How to Contribute: Pull Requests
--------------------------------
We accept pull requests! Thank you for taking the time to read this. There are only a few guidelines before you get started. Make sure you have read the :ref:`developers` and have appropriately setup your project. Please make sure to do the following to appropriately create a pull request for this project.

1. `Fork the project on GitHub <https://github.com/georgianpartners/foreshadow>`_ 
2. Setup the project following the instructions in the :ref:`developers` **using your fork**
3. Create a branch to hold your change

   .. code-block:: console
   
      $ git checkout development
      $ git checkout -b contribution_branch_name

4. Start making changes to this branch and remember to never work on the master branch.
5. Make sure to add tests for your changes to `foreshadow/tests/` and make sure to run those changes. You need to run these commands from the root of the project repository.

   .. code-block:: console

      $ black foreshadow # required formatter
      $ pytest
      $ coverage html
      $ open htmlcov/index.html
      $ tox -r

6. If everything is green and looks good, you're ready to commit

   .. code-block:: console

      $ git add changed_files
      $ git commit # make sure use descriptive commit messages
      $ git push -u origin contribution_branch_name

7. Go to the github fork page and submit your pull request against the **development** branch. Please use the following template for pull requests::

   ###Description
   Please add a narrative description of your the changes made
   ###Related Issue
   Please add any issue that this pull request addresses
   ###Motivation and Context
   If applicable
   ###Screenshots (if appropriate)
