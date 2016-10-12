# Machine Learning: Traffic Sign Classification (Python, 2015)
This was a Python application written for a competition during the Machine Learning course in 2015. It was done in a team of 3. The report can be found under `report.pdf`.

You want to...:
* **Add a new kind of classifier?** Go to classifier.py and extend `BaseClasifier` like with all other classifiers. Usually overriding `__init__` is enough.
* **Add a new type of feature?** Implement the feature in a file in the src/features/ directory, then import that in feature_extraction.py and add it as an `if 'feature_name' in feature_options:` to the `extract_features` function.
_Don't forget to regularly delete the *.pik files in the features/ directory if you are still changing aspects of the feature itself (you don't need to delete it if you are just selecting different features, as the program will detect this and overwrite the files)._
* **Use a different feature/classifier combination?**  Simply add it to model.py and change the `get_model` function to return your model.
