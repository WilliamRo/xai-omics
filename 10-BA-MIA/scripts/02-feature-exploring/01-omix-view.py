from pictor.xomics.omix import Omix



data_path = r'D:/data/BAMIA/BAMIA-Origin-107.omix'
data_path = r'D:\data\BAMIA\CT\rad_features_pool\0-80383-7025732.omix'

omix = Omix.load(data_path)

print(omix.sample_labels)
print(omix.data_name)

# omix.show_in_explorer()
