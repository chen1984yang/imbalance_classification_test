import pandas as pd
from collections import Counter
from os.path import join


def outlier_log_parser(path, dataset="", attr_name="roc"):
	data = pd.read_csv(path)
	n_data = len(data)
	print(n_data)
	method_name_set = {}
	stats = Counter(data["method"])
	methods = list(stats.keys())
	transformed_data = {}
	for m in methods:
		roc = data[data["method"] == m][[attr_name]]
		print(len(roc))
		if len(roc) < 37:
			continue
		transformed_data[m] = roc.values[:37, 0]
		# print(m)
		# print(roc)
	m = list(Counter(data["noise_rate"]).keys())
	print(len(m), m)
	transformed_data["noise_rate"] = m[:37]
	transformed_data = pd.DataFrame(transformed_data)
	transformed_data.to_csv("outlier_{}_{}.csv".format(dataset, attr_name))


if __name__ == '__main__':
	attr_name = "roc"
	dataset = "santander"
	outlier_log_parser(join("tests", "active_modify_{}_test_11.log".format(dataset)), dataset=dataset, attr_name=attr_name)