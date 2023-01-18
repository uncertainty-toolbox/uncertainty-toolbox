Installation
============

Uncertainty Toolbox requires Python 3.6+. For a lightweight installation of the package
only, run:

.. code-block:: console

  pip install uncertainty-toolbox

For a full installation with examples, tests, and the latest updates, run:

.. code-block:: console

  git clone https://github.com/uncertainty-toolbox/uncertainty-toolbox.git
  cd uncertainty-toolbox
  pip install -e . -r requirements/requirements_dev.txt

Note that the previous command requires pip ≥ 21.3.

To verify correct installation, you can run the test suite via:

.. code-block:: console

  source shell/run_all_tests.sh
