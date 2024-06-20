from pictor.xomics.omix import Omix



data_path = r'D:/data/BAMIA/BAMIA-All-851.omix'

omix = Omix.load(data_path)

omix.show_in_explorer()

