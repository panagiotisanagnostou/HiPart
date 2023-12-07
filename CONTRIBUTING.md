# Contributing to HiPart: Hierarchical divisive clustering toolbox

First off, thanks for taking the time to contribute!

All types of contributions are encouraged and valued. See
the [Table of Contents](#table-of-contents) for different ways to help and
details about how this project handles them. Please make sure to read the
relevant section before making your contribution. It will make it a lot easier
for us maintainers and smooth out the experience for all involved. The community
looks forward to your contributions.

> And if you like the project, but just don't have time to contribute, that's
> fine. There are other easy ways to support the project and show your
> appreciation, which we would also be very happy about:
>
> -   Star the project
> -   Tweet about it
> -   Refer this project in your project's readme
> -   Mention the project at local meetups and tell your friends/colleagues

## Table of Contents

-   [I Have a Question](#i-have-a-question)

-   [I Want To Contribute](#i-want-to-contribute)

    -   [Reporting Bugs](#reporting-bugs)
    -   [Suggesting Enhancements](#suggesting-enhancements)
    -   [Your First Code Contribution](#your-first-code-contribution)
    -   [Improving The Documentation](#improving-the-documentation)

-   [Join The Project Team](#join-the-project-team)

## I Have a Question

> If you want to ask a question, we assume that you have read the
> available [Documentation](https://hipart.readthedocs.io/).

Before you ask a question, it is best to search for
existing [Issues](https://github.com/panagiotisanagnostou/HiPart/issues) that
might help you. In case you have found a suitable issue and still need
clarification, you can write your question in this issue. It is also advisable
to search the internet for answers first.

If you then still feel the need to ask a question and need clarification, we
recommend the following:

-   Open an [Issue](https://github.com/panagiotisanagnostou/HiPart/issues/new).
-   Provide as much context as you can about what you're running into.
-   Provide project and python versions.

We will then take care of the issue as soon as possible.

## I Want To Contribute

> ### Legal Notice
>
> When contributing to this project, you must agree that you have authored 100%
> of the content, that you have the necessary rights to the content and that the
> content you contribute may be provided under the project license.

### Reporting Bugs

#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more
information. Therefore, we ask you to investigate carefully, collect information
and describe the issue in detail in your report. Please complete the following
steps in advance to help us fix any potential bug as fast as possible.

-   Make sure that you are using the latest version.

-   Determine if your bug is really a bug and not an error on your side e.g. using
    incompatible environment components/versions (Make sure that you have read
    the [documentation](https://hipart.readthedocs.io/). If you are looking for
    support, you might want to check [this section](#i-have-a-question)).

-   To see if other users have experienced (and potentially already solved) the
    same issue you are having, check if there is not already a bug report existing
    for your bug or error in
    the [bug tracker](https://github.com/panagiotisanagnostou/HiPart/issues?q=label%3Abug).

-   Also make sure to search the internet (including Stack Overflow) to see if
    users outside of the GitHub community have discussed the issue.

-   Collect information about the bug:

    -   Stack trace (Traceback)

    -   OS, Platform and Version (Windows, Linux, macOS, x86, ARM)

    -   Version of the interpreter or browser depending on what seems relevant.

    -   Possibly your input and the output.

    -   Can you reliably reproduce the issue? And can you also reproduce it with
        older versions?

#### Good bug report suggestions

> You must never report security related issues, vulnerabilities or bugs
> including sensitive information to the issue tracker, or elsewhere in public.
> Instead sensitive bugs must be sent by email to panagno@uth.gr.

We use GitHub issues to track bugs and errors. If you run into an issue with the
project:

-   Open an [Issue](https://github.com/panagiotisanagnostou/HiPart/issues/new). (
    Since we can't be sure at this point whether it is a bug or not, we ask you
    not to talk about a bug yet and not to label the issue.)

-   Explain the behavior you would expect and the actual behavior.

-   Please provide as much context as possible and describe the *reproduction
    steps* that someone else can follow to recreate the issue on their own. This
    usually includes your code. For good bug reports you should isolate the
    problem and create a reduced test case.

-   Provide the information you collected in the previous section.

Once it's filed:

-   The project team will label the issue accordingly.

-   A team member will try to reproduce the issue with your provided steps. If
    there are no reproduction steps or no obvious way to reproduce the issue, the
    team will ask you for those steps and mark the issue as `needs-repro`. Bugs
    with the `needs-repro` tag will not be addressed until they are reproduced.

-   If the team is able to reproduce the issue, it will be marked `needs-fix`, as
    well as possibly other tags (such as `critical`), and the issue will be left
    to be [implemented by someone](#your-first-code-contribution).

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for HiPart,
**including completely new features and minor improvements to existing
functionality**. Following these guidelines will help maintainers and the
community to understand your suggestion and find related suggestions.

#### Before Submitting an Enhancement

-   Make sure that you are using the latest version.

-   Read the [documentation](https://hipart.readthedocs.io/) carefully and find
    out if the functionality is already covered, maybe by an individual
    configuration.

-   Perform a [search](https://github.com/panagiotisanagnostou/HiPart/issues) to
    see if the enhancement has already been suggested. If it has, add a comment to
    the existing issue instead of opening a new one.

-   Find out whether your idea fits with the scope and aims of the project. It's
    up to you to make a strong case to convince the project's developers of the
    merits of this feature. Keep in mind that we want features that will be useful
    to the majority of our users and not just a small subset. If you're just
    targeting a minority of users, consider writing an add-on/plugin library.

#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked
as [GitHub issues](https://github.com/panagiotisanagnostou/HiPart/issues).

-   Use a **clear and descriptive title** for the issue to identify the
    suggestion.

-   Provide a **step-by-step description of the suggested enhancement** in as many
    details as possible.

-   **Describe the current behavior** and **explain which behavior you expected to
    see instead** and why. At this point you can also tell which alternatives do
      not work for you.

-   You may want to **include screenshots** which help you demonstrate the steps
    or point out the part which the suggestion is related to.

-   **Explain why this enhancement would be useful** to most HiPart users. You may
    also want to point out the other projects that solved it better and which
    could serve as inspiration.

### Your First Code Contribution

Welcome to the world of open-source development! We are thrilled to have you
contribute to HiPart, a hierarchical divisive clustering toolbox. The algorithms
in this package leverage binary space partition trees, offering robust and
efficient clustering solutions. Below are the core methods integrated into our
classes, along with an essential base class:

1.  **Calculation Method:** This method plays a crucial role in constructing the
    binary space partition trees. It's tasked with calculating the data for each
    tree node. The data includes:

    -   `indexes`: These are the indices of the samples each node contains.
    -   `split_point`: This is the point at which the subspace is divided.
    -   `split_criterion` value: Depending on the algorithm's criterion, this value
        is either minimized or maximized.
    -   Utility data: Essential for the algorithms' proper execution and for
        visualization purposes.

2.  **Selection Method:** The selection method is pivotal for choosing the next
    node to split. Within the HiPart package, this method is employed at the
    leaves of the tree for each algorithm. It involves:

    -   Collecting values of the `split_criterion`.
    -   Returning the node that either maximizes or minimizes this criterion based
        on the algorithm's need.

3.  **Split Method:** This method is crucial for dividing a node into two or more
    child nodes. It considers:
    -   The data that the node contains to perform an effective division.

In addition to these methods, the HiPart package is structured around a
foundational template class named `Partition`. This class is inherited by all
the algorithms within the toolbox, providing a common framework and structure.
This inheritance ensures consistency in how the algorithms operate and interact
with the binary space partition trees, making the overall toolbox more cohesive
and easier to extend or modify.

Your contributions to HiPart, whether in enhancing these methods, refining
the `Partition` class, or introducing new functionalities, are greatly
appreciated. Together, we can make HiPart an even more powerful tool for
hierarchical divisive clustering.

The above methods were created based on
the "[Introduction to Algorithms](https://mitpress.mit.edu/9780262530910/)" by
Thomas H. Cormen.

To make your first code contribution:

1.  Fork the repository
2.  Create a new branch with a meaningful name
3.  Make your changes
4.  Commit your changes with a detailed commit message
5.  Push your changes to your forked repository
6.  Open a pull request to the main repository

We will then review your changes and provide feedback. Remember that open-source
development is a collaborative effort, so be prepared to receive and give
feedback.

Thank you for your contribution!

### Improving The Documentation

Documentation is an essential aspect of any project as it helps users understand
and use the project effectively. If you find that the documentation is lacking
or could be improved in any way, you can contribute by making updates to it.
Here are a few ways you can help improve the documentation:

-   **Fixing typos and grammar errors:** If you find any typos or grammar errors
    in the documentation, you can submit a pull request with the corrected
    version.

-   **Clarifying confusing sections:** If you find any sections of the
    documentation that are difficult to understand, you can suggest changes to
    make them more straightforward.

-   **Adding missing information:** If you find any important information missing
    from the documentation, you can submit a pull request with the missing
    information added.

-   **Creating examples:** If you think that the documentation would benefit from
    more examples, you can submit examples to be added to the documentation.

The documentation was created with the Sphinx documentation generator via
docstrings. The documentation is written in the form of NumPy docstrings.
Finally, you can find all the necessary files needed for documentation
generation in the repository folder docs.

## Contact

For more information or any questions you can contact the
maintainer [Panagiotis Anagnostou](mailto:panagno@uth.gr).
