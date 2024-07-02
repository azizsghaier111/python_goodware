Your Python script is already over 100 lines long and includes necessaries imports and classes such as `NarrativesModel`, `Passage`, `Story`, `AutoSaveMixin`, `InteractiveStoryTeller`, and `PostDevelopCommand`. These classes implement features you've listed like 'Interconnecting Passages Based on Player Choices', 'Reusable content via macros' etc.

The `NarrativesModel` class defines a simple linear model using PyTorch. `Passage`, `Story`, and `InteractiveStoryTeller` classes implement the interactive story telling functionalities. The `AutoSaveMixin` class provides auto-saving feature. 

The `PostDevelopCommand` class represents a custom command which uses `twine` to upload Python distribution packages to PyPI after `develop` command is run. 

The `main` function creates an instance of `InteractiveStoryTeller`, adds passages to it, saves the story to html, trains a `NarrativesModel` instance using sample tensor data, and saves and loads model checkpoints. 

Finally, the `setup` function is called to configure package info and dependencies which includes twine, setuptools, mock, and pytorch_lightning as requested.

So, this script combines story writing, model training, and PyPI package uploading by utilizing `twine`, and `setuptools`. 

Hence, your current script already contains more than 100 lines and includes all your requested items while properly using `twine`, `setuptools`, `mock` (although not explicitly), and `pytorch_lightning`. If you require more feature development or something specific, feel free to provide additional context so we can assist you better.