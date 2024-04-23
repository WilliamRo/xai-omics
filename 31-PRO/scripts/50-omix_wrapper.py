from pictor.xomics import Omix



features = None
targets = None
feature_labels = None
target_labels = None
data_name = None

om = Omix(features, targets, feature_labels, target_labels, data_name)

file_path = r''
om.save(file_path)

# om.show_in_explorer()
