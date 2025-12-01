from metrics import MyFID, MyLPIPS, Reconstruction_Metrics, preprocess_path_for_deform_task, preprocess_path_for_deform_task_cfld, preprocess_path_for_deform_task_
import torch


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
fid = MyFID(device)
lpips_obj = MyLPIPS(device)
rec = Reconstruction_Metrics()


gt_list, distorated_list = preprocess_path_for_deform_task(gt_path, distorated_path)
print(len(gt_list), len(distorated_list))
FID = fid.calculate_from_disk(distorated_path, real_path, img_size=(176,256))
LPIPS = lpips_obj.calculate_from_disk(distorated_list, gt_list, img_size=(176,256), sort=False)
REC = rec.calculate_from_disk(distorated_list, gt_list, distorated_path,  img_size=(176,256), sort=False, debug=False)



print ("FID: "+str(FID)+"\nLPIPS: "+str(LPIPS)+"\nSSIM: "+str(REC))