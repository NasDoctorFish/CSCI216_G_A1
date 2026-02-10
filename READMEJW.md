## Image processing for RAF_DB
1. load image data into BGR2RGB scheme using cv2 (tensor vectors range 0~255, int -> later to float32)
2. permute(switch sequence) of the dimensions, interpolate and normalize by dividing into 255, and (img-self.mean) / selt.std
3. Use DataLoader from torch.utils.data and group the data into batches (batch_size=64, meaning 64 (norm_imgs, label) is in one batch)