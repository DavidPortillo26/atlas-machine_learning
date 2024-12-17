#!/usr/bin/env python3
def preprocess_images(self, images):
    """
    Resizes and rescales the images before processing

    parameters:
        images [list]: images as numpy.ndarrays

    Resizes the images with inter-cubic interpolation
    Rescales the images to have pixel values in the range [0, 1]

    returns:
        tuple of (pimages, image_shapes):
            pimages [numpy.ndarray of shape (ni, input_h, input_w, 3):
                contains all preprocessed images
                ni: number of images preprocessed
                input_h: input height for Darknet model
                input_w: input width for Darknet model
                3: number of color channels
            image_shapes [numpy.ndarray of shape (ni, 2)]:
                contains the original height and width of image
                ni: number of images preprocessed
                2: (image_height, image_width)
    """
    pimages = []
    image_shapes = []
    input_h = self.model.input.shape[1]
    input_w = self.model.input.shape[2]

    for image in images:
        image_shapes.append(image.shape[:2])
        resized = cv2.resize(image, dsize=(input_w, input_h),
                             interpolation=cv2.INTER_CUBIC)
        rescaled = resized / 255.0
        pimages.append(rescaled)
    pimages = np.array(pimages)
    image_shapes = np.array(image_shapes)

    return (pimages, image_shapes)
