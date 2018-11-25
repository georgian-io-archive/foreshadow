.. _contrib:

Contributing
============

Project Setup
-------------
Foreshadow has one main master branch and feature branches if an when collaborative development is required. Each pypi version and their associated commit will be tagged in GitHub. Before each release a new branch will be created freezing that specific version. Pull requests are merged directly into master. This project follows the `semantic versioning`_ standard.

.. _semantic versioning: https://semver.org/

Issues
------
Please feel free to submit issues to github with any bugs you encounter, ideas you may have, or questions about useage. We only ask that you tag them appropriately as *bug fix*, *feature request*, *usage*. Please also follow the following format

.. literalinclude :: ../.github/issue_template.md
   :language: md

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

7. Go to the github fork page and submit your pull request against the **development** branch.  Please use the following template for pull requests

.. literalinclude :: ../.github/pull_request_template.md  
   :language: md
