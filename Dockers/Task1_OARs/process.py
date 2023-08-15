import os
import sys
o_path = os.getcwd()
print(o_path)
sys.path.append(o_path)

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from post_processing import convert_one_hot_label_to_multi_organs
import monai
from monai.data import Dataset, pad_list_data_collate
import monai.transforms as mt
from monai.inferers import sliding_window_inference

import warnings
warnings.filterwarnings('ignore')

class Customalgorithm():
    def __init__(self):
        """
        Do not modify the `self.input_dir` and `self.output_dir`. 
        (Check https://grand-challenge.org/algorithms/interfaces/)
        """
        self.input_dir = "/input/"
        self.output_dir = "/output/images/head-neck-segmentation/"

        """
        Store the validation/test data and predictions into the `self.nii_path` and `self.result_path`, respectively.
        Put your model and pkl files into the `self.weight`.
        """
        self.weight = "./weight/"
        self.nii_path = './temp_path_nii/images'
        self.nii_result_path = './temp_path_nii/result'

        if not os.path.exists(self.nii_path):
            os.makedirs(os.path.join(self.nii_path, 'head-neck-ct'), exist_ok=True)
            os.makedirs(os.path.join(self.nii_path, 'head-neck-contrast-enhanced-ct'), exist_ok=True)
        if not os.path.exists(self.nii_result_path):
            os.makedirs(self.nii_result_path, exist_ok=True)
        self.ct_nii_path = os.path.join(self.nii_path, 'head-neck-ct')
        self.cect_nii_path = os.path.join(self.nii_path, 'head-neck-contrast-enhanced-ct')
        pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):
        img = sitk.ReadImage(mha_input_path)
        sitk.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):
        img = sitk.ReadImage(nii_input_path)
        sitk.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available. Note that the Grand Challenge only has one available GPU.
        """
        print('Checking GPU availability')
        is_available = torch.cuda.is_available()
        print('Available: ' + str(is_available))
        print(f'Device count: {torch.cuda.device_count()}')
        if is_available:
            print(f'Current device: {torch.cuda.current_device()}')
            print('Device name: ' + torch.cuda.get_device_name(0))
            print('Device memory: ' + str(torch.cuda.get_device_properties(0).total_memory))

    def load_inputs(self):      # use two modalities input data
        """
        Read input data (two modalities) from `self.input_dir` (/input/). 
        Please do not modify the path for CT and contrast-CT images.
        """
        ct_mha = os.listdir(os.path.join(self.input_dir, 'images/head-neck-ct/'))[0]
        ctc_mha = os.listdir(os.path.join(self.input_dir, 'images/head-neck-contrast-enhanced-ct/'))[0]
        uuid = os.path.splitext(ct_mha)[0]

        """
        mha-->nii.gz
        """
        self.convert_mha_to_nii(os.path.join(self.input_dir, 'images/head-neck-ct/', ct_mha),
                                os.path.join(self.ct_nii_path, '{}.nii.gz'.format(uuid)))
        self.convert_mha_to_nii(os.path.join(self.input_dir, 'images/head-neck-contrast-enhanced-ct/', ctc_mha),
                                os.path.join(self.cect_nii_path, '{}.nii.gz'.format(uuid)))
        
        # Check the validation/test data exist.
        print('CT:', os.listdir(self.ct_nii_path))
        print('CECT:', os.listdir(self.cect_nii_path))
        return uuid

    def data_transform(self):
        transform = mt.Compose(
            [
                mt.LoadImageD(keys=["image"], image_only=True),
                mt.EnsureChannelFirstD(keys=["image"]),
                mt.OrientationD(keys=["image"], axcodes="RAS"),
                mt.SpacingD(keys=["image"], pixdim=[1.0, 1.0, 4.0], mode=("bilinear")),
                mt.NormalizeIntensityd(keys="image"),
                mt.ToTensorD(keys=["image"]),
            ])
        return transform

    def get_information(self, path):
        nii = sitk.ReadImage(path)
        size = nii.GetSize()
        spacing = nii.GetSpacing()  # [x,y,z]
        origin = nii.GetOrigin()
        direction = nii.GetDirection()
        return size, spacing, origin, direction

    def resampleVolume(self, path, new_size, new_spacing, sampler):
        nii = sitk.ReadImage(path)

        # volumn = sitk.GetArrayFromImage(nii)  # [z,y,x]
        size = nii.GetSize()
        spacing = nii.GetSpacing()
        origin = nii.GetOrigin()
        direction = nii.GetDirection()

        transform = sitk.Transform()
        transform.SetIdentity()

        if new_size == 'None':
            newsize = [0, 0, 0]
            newsize[0] = int(size[0] * spacing[0] / new_spacing[0] + 0.5)
            newsize[1] = int(size[1] * spacing[1] / new_spacing[1] + 0.5)
            newsize[2] = int(size[2] * spacing[2] / new_spacing[2] + 0.5)
        else:
            newsize = new_size
        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(transform)
        resampler.SetInterpolator(sampler)  # data:sitk.sitkBSplineResampler;  label:sitk.sitkNearestNeighbor
        resampler.SetOutputOrigin(origin)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputDirection(direction)
        resampler.SetSize(newsize)
        newvol = resampler.Execute(nii)
        return newvol

    def predict(self, uuid):
        """
        load the model and checkpoint, and generate the predictions. You can replace this part with your own model.
        """

        # Create model ==================================================================================================
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        model = monai.networks.nets.SwinUNETR(
            img_size=(224, 224, 64),
            in_channels=2,
            out_channels=55)
        checkpoint_dir = r'{}/best_metric_model.pth'.format(self.weight)
        checkpoint = torch.load(checkpoint_dir, map_location='cpu')
        model.load_state_dict(checkpoint)
        print('Parameters loading successful!')
        model.to(device)
        model.eval()
        # ===============================================================================================================

        print('Start testing...')
        with torch.no_grad():
            dic_data = [{'image': [os.path.join(self.ct_nii_path, '{}.nii.gz'.format(uuid)),
                                   os.path.join(self.cect_nii_path, '{}.nii.gz'.format(uuid))]}]
            test_ds = Dataset(data=dic_data, transform=self.data_transform())
            test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=pad_list_data_collate)
            for batch_data in test_loader:
                input = batch_data['image'].to(device)

            outputs = sliding_window_inference(input, [224, 224, 64], 1, model)
            if outputs.shape[1] > 1:
                argmax_index = torch.argmax(outputs, dim=1)
                argmax_index = argmax_index.int()
                merged_preds = argmax_index.unsqueeze(dim=1)
            else:
                print('Please ensure that this is multi-label data.')

            output = merged_preds[0, 0, :, :, :].cpu().detach().numpy()
            output = np.flip(output, axis=1)
            output = np.flip(output, axis=0)

            ct_nii = nib.load(os.path.join(self.ct_nii_path, '{}.nii.gz'.format(uuid)))
            new_image = nib.Nifti1Image(output, ct_nii.affine.copy(), ct_nii.header.copy())
            new_image.header.set_zooms((1.0, 1.0, 4.0))
            nib.save(new_image, os.path.join(self.nii_result_path, '{}.nii.gz'.format(uuid)))

            ct_info = self.get_information(os.path.join(self.ct_nii_path, '{}.nii.gz'.format(uuid)))
            n_output = self.resampleVolume(os.path.join(self.nii_result_path, '{}.nii.gz'.format(uuid)), new_size=ct_info[0], new_spacing=ct_info[1], sampler=sitk.sitkNearestNeighbor)
            wriiter = sitk.ImageFileWriter()
            wriiter.SetFileName(os.path.join(self.nii_result_path, '{}.nii.gz'.format(uuid)))
            wriiter.Execute(n_output)
        print('Prediction finished !')

    def write_outputs(self, uuid):
        """
        If you used one-hot label (54 classes) for training, please convert the 54 classes prediction to 45 oars prediction using function `convert_one_hot_label_to_multi_organs`.
        Otherwise, stack your 45 predictions for oars in the first channel, the corresponding mapping between the channel index and the organ names is:
        {0: 'Brain',
        1: 'BrainStem',
        2: 'Chiasm',
        3: 'TemporalLobe_L',
        4: 'TemporalLobe_R',
        5: 'Hippocampus_L',
        6: 'Hippocampus_R',
        7: 'Eye_L',
        8: 'Eye_R',
        9: 'Lens_L',
        10: 'Lens_R',
        11: 'OpticNerve_L',
        12: 'OpticNerve_R',
        13: 'MiddleEar_L',
        14: 'MiddleEar_R',
        15: 'IAC_L',
        16: 'IAC_R',
        17: 'TympanicCavity_L',
        18: 'TympanicCavity_R',
        19: 'VestibulSemi_L',
        20: 'VestibulSemi_R',
        21: 'Cochlea_L',
        22: 'Cochlea_R',
        23: 'ETbone_L',
        24: 'ETbone_R',
        25: 'Pituitary',
        26: 'OralCavity',
        27: 'Mandible_L',
        28: 'Mandible_R',
        29: 'Submandibular_L',
        30: 'Submandibular_R',
        31: 'Parotid_L',
        32: 'Parotid_R',
        33: 'Mastoid_L',
        34: 'Mastoid_R',
        35: 'TMjoint_L',
        36: 'TMjoint_R',
        37: 'SpinalCord',
        38: 'Esophagus',
        39: 'Larynx',
        40: 'Larynx_Glottic',
        41: 'Larynx_Supraglot',
        42: 'PharynxConst',
        43: 'Thyroid',
        44: 'Trachea'}
        Please ensure the 0 channel is the prediction of Brain, the 1 channel is the prediction of BrainStem, ......, the 44 channel is the prediction of Trachea.
        and also ensure the shape of final prediction array is [45, *image_shape].
        The predictions should be saved in the `self.output_dir` (/output/). Please do not modify the path and the suffix (.mha) for saving the prediction.
        """
        os.makedirs(os.path.dirname(self.output_dir), exist_ok=True)
        convert_one_hot_label_to_multi_organs(os.path.join(self.nii_result_path, '{}.nii.gz'.format(uuid)), os.path.join(self.output_dir, uuid + ".mha"))
        print('Output written to: ', os.path.join(self.output_dir, uuid + ".mha"))

    def post_process(self):
        self.check_gpu()
        print('Start processing')
        uuid = self.load_inputs()
        print('Start prediction')
        self.predict(uuid)
        print('Start output writing')
        self.write_outputs(uuid)

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        print(self.weight, self.nii_path, self.nii_result_path)
        self.post_process()


if __name__ == "__main__":
    Customalgorithm().process()
