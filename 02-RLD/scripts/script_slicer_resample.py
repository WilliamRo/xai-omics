import os
import slicer

from tqdm import tqdm
from DICOMLib import DICOMUtils



def loadvolume(patient_path):
  with DICOMUtils.TemporaryDICOMDatabase() as db:
    DICOMUtils.importDicom(patient_path, db)
    DICOMUtils.loadPatientByUID(db.patients()[0])

  loadedNodes = slicer.util.getNodes()
  ct_id, pet_id = None, None
  for nodeID, node in loadedNodes.items():
    if node.IsA("vtkMRMLScalarVolumeNode"):
      if 'CT' in nodeID:
        ct_id = nodeID
      elif 'PET' in nodeID:
        pet_id = nodeID

  assert ct_id is not None
  assert pet_id is not None
  ct = slicer.util.getNode(ct_id)
  ct.SetName('ct')
  pet = slicer.util.getNode(pet_id)
  pet.SetName('pet')

  return ct, pet


if __name__ == '__main__':
  dir = r'E:\xai-omics\data\02-PET-CT-Y1\sg_raw'
  patients = os.listdir(dir)

  for p in tqdm(patients):
    # load the data
    patient_path = os.path.join(dir, p)
    save_path_pet = os.path.join(dir, p, 'pet-resample.nii')
    save_path_ct = os.path.join(dir, p, 'ct.nii')

    ct, pet = loadvolume(patient_path)

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
