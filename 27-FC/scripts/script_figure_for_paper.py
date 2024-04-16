import matplotlib.pyplot as plt
import numpy as np



def draw_boxplot(data_list: list, data_name: list, title):
	assert len(data_list) == len(data_name)

	plt.boxplot(data_list)
	plt.xticks(list(range(len(data_list))), data_name)

	plt.title(title)
	plt.xlabel('Groups')
	plt.ylabel('Values')

	plt.show()


def draw_boxplot_example():
	data1 = np.random.randn(100)
	data2 = np.random.randn(100) + 2
	data3 = np.random.randn(100) - 2
	data4 = np.random.randn(100) * 1.5

	data_list = [data1, data2, data3, data4]
	data_name = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
	title = 'Boxplot Example with 4 Groups'
	draw_boxplot(data_list, data_name, title)



if __name__ == '__main__':
	draw_boxplot_example()