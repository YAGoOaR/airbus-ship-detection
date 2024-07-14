import numpy as np

def encode_RLE(img: np.ndarray) -> str:
    flat_img = np.concatenate([[0], img.T.flatten(), [0]])
    runs = np.where(flat_img[1:] != flat_img[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_RLE(mask_rle: str, shape: tuple[int, int]=(768, 768)) -> np.ndarray:
    image = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    runs = mask_rle.split()

    starts = np.asarray(runs[0:][::2], dtype=int) -1
    lengths = np.asarray(runs[1:][::2], dtype=int)
    ends = starts + lengths

    for s, e in zip(starts, ends):
        image[s:e] = 1

    return image.reshape(shape).T
