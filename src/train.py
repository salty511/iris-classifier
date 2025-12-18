from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import argparse

def train_model(dataset, test_size=0.2, random_state=42):
	"""Train a Decision Tree classifier on the Iris dataset.
	
	Args:
		test_size: Proportion of the dataset to include in the test split
		random_state: Random state for reproducibility
		
	Returns:
		tuple: (model, X_test, y_test, y_pred, iris)
	"""

	X = dataset.data
	y = dataset.target

	# Split the dataset into training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

	# Train a Decision Tree classifier
	model = DecisionTreeClassifier(random_state=random_state)
	model.fit(X_train, y_train)

	# Make predictions on the test set
	y_pred = model.predict(X_test)
	
	return model, X_test, y_test, y_pred


def print_metrics(y_test, y_pred, model, feature_names):

	from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
	import matplotlib.pyplot as plt
	from sklearn.tree import plot_tree
	import os

	accuracy = accuracy_score(y_test, y_pred)
	print("Accuracy: ", accuracy)

	report = classification_report(y_test, y_pred)
	print(report)

	confusion = confusion_matrix(y_test, y_pred)
	disp = ConfusionMatrixDisplay(confusion)
	disp.plot()

	if not os.path.exists("outputs"):
		os.makedirs("outputs")
		
	plt.savefig("outputs/confusion_matrix.png")

	print("Feature importances:")
	for i in range(len(model.feature_importances_)):
		print(f'{feature_names[i]}: {model.feature_importances_[i]}')

	plot_tree(model)


if __name__ == '__main__':
	# Parse command-line arguments
	parser = argparse.ArgumentParser(description='Train a Decision Tree classifier on the Iris dataset')
	parser.add_argument('--test-size', type=float, default=0.2,
	                    help='Proportion of the dataset to include in the test split (default: 0.2)')
	parser.add_argument('--random-state', type=int, default=42,
	                    help='Random state for reproducibility (default: 42)')
	args = parser.parse_args()

	# Load the Iris dataset
	iris = load_iris()

	# Train the model
	model, X_test, y_test, y_pred = train_model(
		iris,
		test_size=args.test_size,
		random_state=args.random_state
	)
	
	# Print the metrics
	print_metrics(y_test, y_pred, model, iris.feature_names)
