import matplotlib.pyplot as plt
import argparse

def pad(arr, desired_len):
	start = len(arr)
	for i in range(desired_len - start):
		arr.append(0)

def plot_graphs(experiment_names):
	for experiment_name in experiment_names:
		with open("./experiments/"+experiment_name +  "/train_loss_every_10.txt") as file:
			loss = file.readlines()
			loss = [float(x) for x in loss]
		plt.plot(loss)
	plt.title("Smoothed Loss")
	plt.ylabel('smoothed loss')
	plt.xlabel('Iterations')
	plt.legend(experiment_names)
	plt.show()

	for experiment_name in experiment_names:
		print experiment_name
		with open("./experiments/"+experiment_name +  "/dev_loss.txt") as file:
			loss = file.readlines()
			iteration = [int(x.split("\t")[0]) for x in loss]
			loss = [float(x.split("\t")[1]) for x in loss]
		plt.plot(iteration, loss)

	plt.legend(experiment_names)

	plt.title("Dev Loss")
	plt.ylabel('Dev loss')
	plt.xlabel('Iterations')

	plt.show()

	for experiment_name in experiment_names:

		with open("./experiments/"+experiment_name +  "/F1_loss.txt") as file:
			loss = file.readlines()
			iteration = [int(x.split("\t")[0]) for x in loss]
			loss = [float(x.split("\t")[1]) for x in loss]

		plt.plot(iteration, loss)
	plt.legend(experiment_names)

	plt.title("F1 Score")
	plt.ylabel('F1 Score')
	plt.xlabel('Iterations')

	plt.show()


	for experiment_name in experiment_names:

		with open("./experiments/"+experiment_name +  "/EM_loss.txt") as file:
			loss = file.readlines()
			iteration = [int(x.split("\t")[0]) for x in loss]
			loss = [float(x.split("\t")[1]) for x in loss]
		plt.plot(iteration, loss)

	plt.legend(experiment_names)

	plt.title("EM Score")
	plt.ylabel('EM Score')
	plt.xlabel('Iterations')
	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process some integers.')

	parser.add_argument('--experiment_names', default="bidaf_two_layer", type=str, nargs='+', help='sum the integers (default: find the max)')
	args = parser.parse_args()
	plot_graphs(args.experiment_names)