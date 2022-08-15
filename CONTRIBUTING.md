# Contributing

We welcome and greatly appreciate contributions from the community! You can
help make the Uncertainty Toolbox even better by either

1. [Creating a PR](#create-a-pr) to fix a bug or add functionality.
2. [Reporting a bug](#report-a-bug) that you found in the toolbox.
3. [Requesting a feature](#request-a-feature) that you would like to
   see in the toolbox.

# Create a PR

To create a PR:

1. If applicable, make sure to write unit tests that exhaustively test your
   added code. Run all unit tests, using
   `source shell/run_all_tests.sh`, and ensure that there are no failures.
2. Please add docstrings (following the [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)) 
   and type hinting ([example here](https://github.com/uncertainty-toolbox/uncertainty-toolbox/blob/946433b2bca9eb93b06b144cffdb32faf0a9c64f/uncertainty_toolbox/metrics.py#L242)).
   Also, format all code using the [black format](https://black.readthedocs.io/en/stable/) 
   by running `source shell/format_black.sh`.
3. Submit a PR to the [main branch](https://github.com/uncertainty-toolbox/uncertainty-toolbox/tree/main).
   If your PR fixes a bug, detail what the problem was and how it was fixed.
   If your PR adds code, include justification for why this code should be added.
4. Please allow maintainers to edit your pull request by following [these instructions](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork).
5. The maintainers will discuss your PR and merge it into the main branch if
   accepted.

### What kind of additional features are we looking for?

If there is an important metric that you believe is missing from Uncertainty
Toolbox, we would be grateful if you submitted a PR adding this metric. For
other types of features, ask yourself:

* Does this change add a new feature that will have a positive effect on the
  majority of the toolbox's users?
* Does this change make the codebase more confusing or difficult to deal with?
* Does this change add heavy or niche dependencies?

Feel free to [submit an issue](https://github.com/uncertainty-toolbox/uncertainty-toolbox/issues) 
before making the feature to see if your feature is something that the 
maintainers or other users would want.

# Report a Bug

If you find a bug, please make an issue and include [Bug] in the title. In 
your issue, please give a description of how to reproduce the bug.

# Request a Feature

For any feature request, submit an issue with [Feature Request] in the title.
As part of the issue, describe why this would be beneficial for the toolbox and
give an example use case of how the feature would be used.

# Maintainers

If you have any questions, feel free to reach out to the maintainers:

* [Youngseog Chung](https://github.com/YoungseogChung) (youngsec (at) cs.cmu.edu)
* [Willie Neiswanger](https://github.com/willieneis) (neiswanger (at) cs.stanford.edu)
* [Ian Char](https://github.com/IanChar) (ichar (at) cs.cmu.edu)
* [Han Guo](https://github.com/HanGuo97) (hanguo (at) cs.cmu.edu)
