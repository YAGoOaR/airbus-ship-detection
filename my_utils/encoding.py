import numpy as np
import cv2

def encode_RLE(img: np.ndarray) -> str:
    '''
    Encode segmentation masks to RLE (Run-length encoding).
    '''
    # We can represent flat image as f(x) where x is pixel index 
    f_x = np.concatenate([[0], img.T.flatten()]) # f(x)
    f_x_plus_1 = np.concatenate([img.T.flatten(), [0]]) # f(x+1)

    deltas, _ = np.where(f_x != f_x_plus_1) # get pixel indices where mask value changes
    runs = deltas + 1 # shift numeration
    runs[1::2] -= runs[::2] # subtract run starts from ends (get run lengths)
    return ' '.join(str(x) for x in runs) # join into a string

def decode_RLE(
        mask_rle: str,
        encoding_shape: tuple[int, int] = (768, 768),
        out_shape: tuple[int, int] = (768, 768),
    ) -> np.ndarray:
    '''
    Decode RLE into images.
    '''
    image = np.zeros(encoding_shape[0] * encoding_shape[1], dtype=np.uint8) # empty mask
    
    # Parse starts and lengths, get run ends
    runs = mask_rle.split()
    starts = np.asarray(runs[0:][::2], dtype=int) - 1
    lengths = np.asarray(runs[1:][::2], dtype=int)
    ends = starts + lengths

    # for each run, draw corresponding pixels in the mask
    for s, e in zip(starts, ends):
        image[s:e] = 1

    # Get desired shape
    image = image.reshape(encoding_shape).T
    return cv2.resize(image, out_shape)
