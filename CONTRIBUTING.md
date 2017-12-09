# On Github Issues and Pull Requests

Found a bug? Have a new feature to suggest? Want to contribute changes to the codebase? Make sure to read this first.

## Bug reporting

Your code doesn't work, and you have determined that the issue lies with Keras-VGG16-places365? Follow these steps to report a bug.

1. Your bug may already be fixed. Make sure to update to the current Keras master branch (``` pip install git+git://github.com/fchollet/keras.git --upgrade --no-deps ```), as well as the latest Theano/TensorFlow/CNTK master branch.
To easily update Theano: `pip install git+git://github.com/Theano/Theano.git --upgrade`

2. Search for similar issues. It's possible somebody has encountered this bug already. Also remember to check out Keras' [FAQ](http://keras.io/faq/). Still having a problem? Open an issue on Github to let us know.

3. Make sure you provide us with useful information about your configuration: what OS are you using? What Keras backend are you using? Are you running on GPU?

4. Provide us with a script to reproduce the issue. This script should be runnable as-is and should not require external data download (use randomly generated data if you need to run a model on some test data). We recommend that you use Github Gists to post your code.

5. If possible, take a stab at fixing the bug yourself --if you can!

The more information you provide, the easier it is for us to validate that there is a bug and the faster we'll be able to take action. If you want your issue to be resolved quickly, following the steps above is crucial.

---

## Requesting a Feature

You can also use Github issues to request features you would like to see in Keras-VGG16-places365.

1. Provide a clear and detailed explanation of the feature you want and why it's important to add. 

2. Provide code snippets demonstrating the API you have in mind and illustrating the use cases of your feature. Of course, you don't need to write any real code at this point!

3. After discussing the feature you may choose to attempt a Pull Request. If you're at all able, start writing some code. We always have more work to do than time to do it. If you can write some code then that will speed the process along.


---

## Pull Requests

**Where should I submit my pull request?**

1. **Keras-VGG16-places365 improvements and bugfixes** go to the [Keras-VGG16-places365 `master` branch](https://github.com/GKalliatakis/Keras-VGG16-places365/tree/master).


Here's a quick guide to submitting your improvements:


1. Write the code (or get others to write it). This is the hard part!

2. When committing, use appropriate, descriptive commit messages.

3. Update the documentation. If introducing new functionality, make sure you include code snippets demonstrating the usage of your new feature.

4. Submit your PR. If your changes have been approved in a previous discussion, your PR is likely to be merged promptly.

---

## Adding new examples

Even if you don't contribute to the Keras-VGG16-places365 source code, if you have an application of Keras-VGG16-places365 that is concise and powerful, please consider adding it to our collection of examples. 
