import numpy as np

def detect_blur_fft(image, size=60, thresh=10):
	# grab the dimensions of the image and use the dimensions to
	# derive the center (x, y)-coordinates
	(h, w) = image.shape
	(cX, cY) = (int(w / 2.0), int(h / 2.0))
	fft = np.fft.fft2(image)
	fftShift = np.fft.fftshift(fft)
	fftShift[cY - size:cY + size, cX - size:cX + size] = 0
	fftShift = np.fft.ifftshift(fftShift)
	recon = np.fft.ifft2(fftShift)
	magnitude = 20 * np.log(np.abs(recon))
	mean = np.mean(magnitude)
	# the image will be considered "blurry" if the mean value of the
	# magnitudes is less than the threshold value
	return mean, mean <= thresh

def get_less_blurry_image(images):
    means = np.array([])
    for image in images:
        mean, _ = detect_blur_fft(image)
        means = np.array(means, mean)
    return np.argmax(means)
