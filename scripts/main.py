import os
import sys
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
sys.path.append('.')

from training.config import get_config
from training.inference import Inference
from training.utils import create_logger, print_args

# python scripts/main.py transfer --input assets/images/non-makeup/source_1.png --output result/resut.png --ref assets/images/makeup/reference_1.png
# python scripts/main.py transfer_all --reference-dir ../Makeups
# python scripts/main.py transfer_all --reference-dir ../Makeups --source-dir ../Sources/Small --save_path ../Results/elegant_2
# python scripts/main.py transfer_all --reference-dir ../Makeups --source-dir ../Sources/Small --save_path ../Results/elegant_3 --mask_area lips

def main(config, args):
    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    # logger.info(config)

    inference = Inference(config, args, args.load_path)

    command = args.command
    
    if command == "transfer":
        if args.input:
            imgA = Image.open(args.input).convert('RGB')
            imgB = Image.open(args.ref).convert('RGB')
            output_dir = os.path.split(args.output)[0]
            file_name = os.path.split(args.input)[1].split('.')[0]
            diff_img_path = os.path.join(output_dir, f"diff_{file_name}.png")
            if args.mask_area and args.mask_area != "":
                mask_area = args.mask_area.split(',')
                logger.info(f"Perform selective transferring: {args.mask_area}")
                result = inference.transfer_selective(imgA, imgB, mask_area, diff_img_path, postprocess=True)
            else:
                logger.info(f"Perform standard transferring")
                result = inference.transfer(imgA, imgB, diff_img_path, postprocess=True) 
            
            if result:
                result = np.array(result)
                Image.fromarray(result.astype(np.uint8)).save(args.output)
    else:
        n_imgname = sorted(os.listdir(args.source_dir))
        m_imgname = sorted(os.listdir(args.reference_dir))

        for i, imga_name in enumerate(n_imgname):
            if imga_name == ".DS_Store":
                continue

            imgA = Image.open(os.path.join(args.source_dir, imga_name)).convert('RGB')
            for j, imgb_name in enumerate(m_imgname):
                if os.path.isdir(imgb_name) or imgb_name == ".DS_Store":
                    continue
                imgB = Image.open(os.path.join(args.reference_dir, imgb_name)).convert('RGB')

                save_path = os.path.join(args.save_folder, f"result_{i}_{j}.png")
                diff_img_path = os.path.join(args.save_folder, f"diff_{i}_{j}.png")
                print(f"transfer: source {imga_name}, ref {imgb_name} -> {save_path}")

                if args.mask_area:
                    mask_area = args.mask_area.split(',')
                    result = inference.transfer_selective(imgA, imgB, mask_area, diff_img_path=diff_img_path, postprocess=True) 
                else:
                    result = inference.transfer(imgA, imgB, diff_img_path=diff_img_path, postprocess=True) 
                
                if result:
                    result = np.array(result)
                    Image.fromarray(result.astype(np.uint8)).save(save_path)

def transfer(inference: Inference, source: Image, ref: Image, result_path: str):
    imgA = Image.open(source).convert('RGB')
    imgB = Image.open(ref).convert('RGB')
    result = inference.transfer(imgA, imgB, postprocess=True) 
    result = np.array(result)
    Image.fromarray(result.astype(np.uint8)).save(result_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for run")
    parser.add_argument("command", type=str, default='transfer')
    parser.add_argument("--name", type=str, default='demo')
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--ref", type=str)
    parser.add_argument("--mask_area", type=str, required=False)
    parser.add_argument("--no_face_cropping", default=False, action="store_true")
    parser.add_argument("--comp_result", default=False, action="store_true")
    parser.add_argument("--save_path", type=str, default='result', help="path to save model")
    parser.add_argument("--load_path", type=str, help="folder to load model", 
                        default='ckpts/sow_pyramid_a5_e3d2_remapped.pth')

    parser.add_argument("--source-dir", type=str, default="assets/images/non-makeup")
    parser.add_argument("--reference-dir", type=str, default="assets/images/makeup")
    parser.add_argument("--gpu", default='0', type=str, help="GPU id to use.")

    args = parser.parse_args()
    args.gpu = 'cuda:' + args.gpu
    # args.device = torch.device(args.gpu)
    args.device = torch.device("mps")

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    config = get_config()
    main(config, args)