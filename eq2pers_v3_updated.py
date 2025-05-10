import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
import os


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def equi2pers(erp_img, fov, patch_size):
    bs, _, erp_h, erp_w = erp_img.shape
    height, width = pair(patch_size)
    fov_h, fov_w = pair(fov)
    FOV = torch.tensor([fov_w/360.0, fov_h/180.0], dtype=torch.float32)

    PI = math.pi
    PI_2 = math.pi * 0.5
    PI2 = math.pi * 2
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width))
    screen_points = torch.stack([xx.flatten(), yy.flatten()], -1)

    # num_rows = 3
    # num_cols = [3, 4, 3]
    # phi_centers = [-60, 0, 60]
    num_rows = 4
    num_cols = [3, 6, 6, 3]
    phi_centers = [-67.5, -22.5, 22.5, 67.5]
    phi_interval = 180 // num_rows
    all_combos = []
    erp_mask = []
    for i, n_cols in enumerate(num_cols):
        for j in np.arange(n_cols):
            theta_interval = 360 / n_cols
            theta_center = j * theta_interval + theta_interval / 2

            center = [theta_center, phi_centers[i]]
            all_combos.append(center)
            up = phi_centers[i] + phi_interval / 2
            down = phi_centers[i] - phi_interval / 2
            left = theta_center - theta_interval / 2
            right = theta_center + theta_interval / 2
            up = int((up + 90) / 180 * erp_h)
            down = int((down + 90) / 180 * erp_h)
            left = int(left / 360 * erp_w)
            right = int(right / 360 * erp_w)
            mask = np.zeros((erp_h, erp_w), dtype=int)
            mask[down:up, left:right] = 1
            erp_mask.append(mask)
    all_combos = np.vstack(all_combos) 
    shifts = np.arange(all_combos.shape[0]) * width
    shifts = torch.from_numpy(shifts).float()
    erp_mask = np.stack(erp_mask)
    erp_mask = torch.from_numpy(erp_mask).float()
    num_patch = all_combos.shape[0]

    center_point = torch.from_numpy(all_combos).float()  # -180 to 180, -90 to 90
    center_point[:, 0] = (center_point[:, 0]) / 360  #0 to 1
    center_point[:, 1] = (center_point[:, 1] + 90) / 180  #0 to 1

    cp = center_point * 2 - 1
    cp[:, 0] = cp[:, 0] * PI
    cp[:, 1] = cp[:, 1] * PI_2
    cp = cp.unsqueeze(1)
    convertedCoord = screen_points * 2 - 1
    convertedCoord[:, 0] = convertedCoord[:, 0] * PI
    convertedCoord[:, 1] = convertedCoord[:, 1] * PI_2
    convertedCoord = convertedCoord * (torch.ones(screen_points.shape, dtype=torch.float32) * FOV)
    convertedCoord = convertedCoord.unsqueeze(0).repeat(cp.shape[0], 1, 1)

    x = convertedCoord[:, :, 0]
    y = convertedCoord[:, :, 1]

    rou = torch.sqrt(x ** 2 + y ** 2)
    c = torch.atan(rou)
    sin_c = torch.sin(c)
    cos_c = torch.cos(c)
    lat = torch.asin(cos_c * torch.sin(cp[:, :, 1]) + (y * sin_c * torch.cos(cp[:, :, 1])) / rou)
    lon = cp[:, :, 0] + torch.atan2(x * sin_c, rou * torch.cos(cp[:, :, 1]) * cos_c - y * torch.sin(cp[:, :, 1]) * sin_c)
    lat_new = lat / PI_2 
    lon_new = lon / PI 
    lon_new[lon_new > 1] -= 2
    lon_new[lon_new<-1] += 2 

    lon_new = lon_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, num_patch*width)
    lat_new = lat_new.view(1, num_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, num_patch*width)
    grid = torch.stack([lon_new, lat_new], -1)
    grid = grid.unsqueeze(0).repeat(bs, 1, 1, 1).to(erp_img.device)

    pers = F.grid_sample(erp_img, grid, mode='bilinear', padding_mode='border', align_corners=True)

    return pers


def process_dataset(dataset_folder, output_folder):
    print(os.walk(dataset_folder))
    for root, _, files in os.walk(dataset_folder):
        print(files)
        for file in sorted(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                img_new = img.astype(np.float32)
                img_new = np.transpose(img_new, [2, 0, 1])
                img_new = torch.from_numpy(img_new)
                img_new = img_new.unsqueeze(0)
                patch_size = 192
                
                pers = equi2pers(img_new, fov=(80, 80), patch_size=(patch_size, patch_size)) # todo: change
                pers = pers[0].numpy()
                pers = pers.transpose(1, 2, 0).astype(np.uint8)

                # saving tangents
                # save the stitched image as .png
                
                pers_size = pers.shape[0]
                total_images = pers.shape[1] // pers_size
                img_width = pers_size
                # sub_output_folder = os.path.join(output_folder, os.path.splitext(file)[0])
                # for i in range(total_images):
                #     # extract and save each sub-image
                #     sub_img = pers[:, i * img_width: (i + 1) * img_width]
                #     img_name = os.path.join(sub_output_folder, f'512_18_image_{i + 1}.jpg')
                #     cv2.imwrite(img_name, sub_img)
                #     print(f'saved {img_name}')

                #######
                
                pers_grid = np.zeros((patch_size * 3, patch_size * 6, 3), dtype=np.uint8)
                #pers_grid = np.zeros((256 * 3, 256 * 4, 3), dtype=np.uint8)
                order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
                # order = [0, 1, 2, 15, 16, 17, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                #order = [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9]
                for i, idx in enumerate(order):
                    row, col = divmod(i, 6)
                    pers_grid[row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size] = pers[:, idx * patch_size:(idx + 1) * patch_size]
                
                sub_output_folder = os.path.join(output_folder, os.path.splitext(file)[0])
                # #os.makedirs(sub_output_folder, exist_ok=True)
                pers_path = os.path.join(sub_output_folder, '18natural_576_secondversion.jpg')
                pers_grid = cv2.resize(pers_grid, (patch_size*6, patch_size*3))
                cv2.imwrite(pers_path, pers_grid)
                print(f'Saved {pers_path}')

def process_image(file_path, output_dir):
    img = cv2.imread(file_path, cv2.imread_color)
    img_new = img.astype(np.float32)
    img_new = np.transpose(img_new, [2, 0, 1])
    img_new = torch.from_numpy(img_new)
    img_new = img_new.unsqueeze(0)
    
    # change field of view (fov) and patch size as needed
    pers = equi2pers(img_new, fov=(80, 80), patch_size=(256, 256))
    
    # convert the output tensor to numpy array
    pers = pers[0].numpy()
    pers = pers.transpose(1, 2, 0).astype(np.uint8)

    # save the stitched image as .png
    pers_path = os.path.join(output_dir, 'pers.png')
    cv2.imwrite(pers_path, resized_pers)

    # remove all .jpg files in the output directory
    for file in os.listdir(output_dir):
        if file.endswith('.jpg'):
            os.remove(os.path.join(output_dir, file))
    
    pers_size = pers.shape[0]
    total_images = pers.shape[1] // pers_size
    img_width = pers_size
    
    for i in range(total_images):
        # extract and save each sub-image
        sub_img = pers[:, i * img_width: (i + 1) * img_width]
        img_name = os.path.join(output_dir, f'image_{i + 1}.jpg')
        cv2.imwrite(img_name, sub_img)
        print(f'saved {img_name}')

def process_image_input(image, patch_size=256):
    image = image.unsqueeze(0)
    pers = equi2pers(image, fov=(80, 80), patch_size=(patch_size, patch_size))
    pers = pers[0]
    tangent_images = []
    # print(pers.shape)
    for i in range(18):
        sub_img = pers[:, :, i * patch_size: (i+ 1) * patch_size ]
        tangent_images.append(sub_img)
    tangent_images = torch.stack(tangent_images, dim=0)
    return tangent_images

if __name__ == '__main__':
    dataset_folder = '/home/hcapuk20/datasets/flickr360/HR'
    output_folder = '/home/hcapuk20/datasets/flickr360/18_tangent_grids'
    process_dataset(dataset_folder, output_folder)
    # erp_img = torch.randn(1, 3, 512, 1024)  # example ERP image
    # fov = (80, 80)
    # patch_size = 256

    # pers, overlap_map = equi2pers(erp_img, fov, patch_size)

    # # Normalize for visualization
    # overlap_np = overlap_map.cpu().numpy()
    # overlap_vis = (overlap_np / overlap_np.max()) * 255
    # overlap_vis = overlap_vis.astype(np.uint8)

    # # Save as colormap
    # colored = cv2.applyColorMap(overlap_vis, cv2.COLORMAP_JET)
    # cv2.imwrite("./overlap_map_equi2pers.png", colored)