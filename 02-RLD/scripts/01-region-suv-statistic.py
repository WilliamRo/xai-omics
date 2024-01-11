import os
import slicer

from tqdm import tqdm
from DICOMLib import DICOMUtils



def loadvolume(patient_path):
  with DICOMUtils.TemporaryDICOMDatabase() as db:
    DICOMUtils.importDicom(patient_path, db)
    DICOMUtils.loadPatientByUID(db.patients()[0])

  loadedNodes = slicer.util.getNodes()
  pet_id = None
  for nodeID, node in loadedNodes.items():
    if node.IsA("vtkMRMLScalarVolumeNode"):
      if 'CT' in nodeID:
        ct_id = nodeID
      if 'PET' in nodeID:
        pet_id = nodeID

  assert ct_id is not None
  assert pet_id is not None
  ct = slicer.util.getNode(ct_id)
  ct.SetName('ct')
  pet = slicer.util.getNode(pet_id)
  pet.SetName('pet')

  return ct, pet


if __name__ == '__main__':
  dir = r'D:/projects/xai-omics/data/02-RLD-RAW/'
  patients = os.listdir(dir)

  for p in tqdm(patients):
    # load the data
    patient_path = os.path.join(dir, p)
    patient_path = os.path.join(os.listdir(patient_path)[0], patient_path)
    image_path = os.listdir(patient_path)
    save_path_pet = os.path.join(patient_path)

    for path in image_path:
      real_path = os.path.join(patient_path, path)
      pet = loadvolume(real_path)
      # resample
      outputVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
      outputVolumeNode.SetName('output')

      resampleModule = slicer.modules.resamplescalarvectordwivolume

      parameters = {}
      parameters['inputVolume'] = slicer.util.getNode('pet')
      parameters['outputVolume'] = slicer.util.getNode('output')
      parameters['referenceVolume'] = slicer.util.getNode('ct')
      parameters['interpolationType'] = 'linear'

      cliNode = slicer.cli.runSync(resampleModule, None, parameters)

    # save the data
    slicer.util.saveNode(slicer.util.getNode('output'), save_path_pet)
    slicer.util.saveNode(slicer.util.getNode('ct'), save_path_ct)

    # delete all nodes to free space
    imageNodes = slicer.mrmlScene.GetNodesByClass('vtkMRMLScalarVolumeNode')

    for node in imageNodes:
      slicer.mrmlScene.RemoveNode(node)

    print(p, 'is ok!\n')
