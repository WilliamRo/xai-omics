from pictor.xomics.omix import Omix
from roma import finder



# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
pool_dir = r'D:/data/BAMIA/CT/rad_features_pool'

# -----------------------------------------------------------------------------
# Read all omix and merge
# -----------------------------------------------------------------------------
omix_list = [Omix.load(fp) for fp in finder.walk(pool_dir, pattern='*.omix')]
omix = Omix.sum(omix_list, data_name='BAMIA-Radomics-111x851')
omix.show_in_explorer()

# use `:save` command to save .omix file
