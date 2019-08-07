# Recommenders 

This [MLHub](https://mlhub.ai) package provides an overview of the
[Microsoft Recommenders](https://github.com/microsoft/recommenders)
repository on github which in turn provides examples and best
practices for building recommendation systems. This MLHub package
provides a demonstration of using the repository, including a
demonstration of the smart adaptive recommender (SAR) and restricted
Boltzmann machine (RBM) algorithms for building recommendation
engines.

The hard work is done using the utilities provided in
[reco_utils](https://github.com/microsoft/recommenders/tree/master/reco_utils)
to support common tasks such as loading datasets in the format
expected by different algorithms, evaluating model outputs, and
splitting train/test data. Implementations of several state-of-the-art
algorithms are provided for self-study and customization in your own
applications.

The MovieLens data sets are used in this demonstration, containing
100,004 ratings across 9125 movies created by 671 users between
9 January 1995 and 16 October 2016. The dataset records the userId,
movieId, rating, timestamp, title, and genres. The goal is to build a
recommendation model to recommend new movies to users.

Visit the github repository for more details:
<https://github.com/microsoft/recsysbp>


## Usage

- To install mlhub (Ubuntu 18.04 LTS)

```console
$ pip3 install mlhub
```

- To install and configure the demo:

```console
$ ml install   recsysbp
$ ml configure recsysbp
```

## Demonstration



# Contributing

This project welcomes contributions and suggestions.  Most
contributions require you to agree to a Contributor License Agreement
(CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine
whether you need to provide a CLA and decorate the PR appropriately
(e.g., label, comment). Simply follow the instructions provided by the
bot. You will only need to do this once across all repos using our
CLA.

This project has adopted the [Microsoft Open Source Code of
Conduct](https://opensource.microsoft.com/codeofconduct/).  For more
information see the [Code of Conduct
FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact
[opencode@microsoft.com](mailto:opencode@microsoft.com) with any
additional questions or comments.

# Legal Notices

Microsoft and any contributors grant you a license to the Microsoft
documentation and other content in this repository under the [Creative
Commons Attribution 4.0 International Public
License](https://creativecommons.org/licenses/by/4.0/legalcode), see
the [LICENSE](LICENSE) file, and grant you a license to any code in
the repository under the [MIT
License](https://opensource.org/licenses/MIT), see the
[LICENSE-CODE](LICENSE-CODE) file.

Microsoft, Windows, Microsoft Azure and/or other Microsoft products
and services referenced in the documentation may be either trademarks
or registered trademarks of Microsoft in the United States and/or
other countries.  The licenses for this project do not grant you
rights to use any Microsoft names, logos, or trademarks.  Microsoft's
general trademark guidelines can be found at
http://go.microsoft.com/fwlink/?LinkID=254653.

Privacy information can be found at
https://privacy.microsoft.com/en-us/

Microsoft and any contributors reserve all other rights, whether under
their respective copyrights, patents, or trademarks, whether by
implication, estoppel or otherwise.
