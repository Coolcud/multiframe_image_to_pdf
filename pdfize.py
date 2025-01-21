import argparse
import logging
import logging.handlers
import os
import sys
from multiprocessing import Pool, cpu_count
from typing import List

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output
from tqdm import tqdm

# TODO: All specified tiffs should be put into one final pdf. One program run => One pdf output.
# TODO: Update image preview to preserve aspect ratio
# TODO: Make hough enablement the default, with a flag to --disable_hough.
# TODO: Improve documentation of the deskew function
# TODO: Update the readme.

def parse_arguments() -> argparse.Namespace:
    """Argument parser for user input."""

    parser = argparse.ArgumentParser(
        description="Deskew and convert multi-frame images to PDF."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where the deskewed PDF files will be saved.",
    )
    parser.add_argument(
        "image_paths",
        nargs="+",
        type=str,
        help="One or more multi-frame image paths to deskew and convert.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) Default is INFO.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug previews of images.",
    )
    parser.add_argument(
        "--use_hough",
        action="store_true",
        help="Use Hough Line Transform for additional skew detection.",
    )
    parser.add_argument(
        "--resize_factor",
        type=float,
        default=1.0,
        help="Resize factor for images before saving to PDF (e.g. 0.5 for half). Default is 1.0 (no resize).",
    )

    return parser.parse_args()


def setup_logging(log_level: str, name: str = None):
    """Log information for user based on preferences."""

    try:
        format = "[%(asctime)s|%(levelname)s]: %(message)s"
        if str is not None:
            format = f"[%(asctime)s|%(levelname)s|{name}]: %(message)s"

        logging.basicConfig(
            level=log_level.upper(),
            format=format,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    except ValueError:
        raise ValueError(f"'{log_level}' is not a valid logging Level!")


def worker_process(
    path: str,
    log_level: str,
    output_dir: str,
    debug: bool = False,
    use_hough: bool = False,
    resize_factor: float = 1.0,
):
    setup_logging(log_level, os.path.basename(path))
    logging.info("New worker started.")

    convert_single_image(path, output_dir, debug, True, use_hough, resize_factor)


def get_tesseract_path() -> str:
    """Return the path to the Tesseract executable."""

    return "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


def read_multiframe_image(file_path: str) -> List[Image.Image]:
    """Return a list of PIL.Image objects from a multi-frame image file."""

    try:
        image = Image.open(file_path)
    except IOError:
        logging.error(f"Unable to open {file_path}.")
        return None

    extracted_images = []
    try:
        i = 0
        while True:
            # Seek frame 'i' (For image formats like .tif with multiple images, get each individually)
            image.seek(i)
            extracted_images.append(image.copy())
            i += 1
    except EOFError:
        pass

    logging.info(f"Extracted {len(extracted_images)} image frames.")
    return extracted_images


def log_image_info(pil_image: Image.Image):
    """Log information about the PIL.Image object."""

    logging.debug(f"Image file format: {pil_image.format}")
    logging.debug(f"Image pixel format: {pil_image.mode}")
    logging.debug(f"Image size (width, height): {pil_image.size}")
    logging.debug(f"Image palette: {pil_image.palette}\n")


def get_skew_angle_tesseract(pil_image: Image.Image) -> float:
    """
    Use Tesseract OCR to determine the skew angle of the image.
    Returns the angle in degrees.
    """

    try:
        osd = pytesseract.image_to_osd(pil_image, lang="lat", output_type=Output.DICT)
        angle = osd["orientation"]
        script = osd["script"]
        confidence = osd["orientation_conf"]
        logging.info(f"Tesseract found {script} at angle: {angle}, conf: {confidence}")

        if script != "Latin":
            logging.warning(f"Ignoring tesseract angle as script was {script}")
            return None

        if confidence < 1:
            logging.warning(f"Ignoring tesseract as confidence was {str(confidence)}")
            return None

        match int(angle):
            case 0:
                return None
            case 90:
                return cv2.ROTATE_90_COUNTERCLOCKWISE
            case 180:
                return cv2.ROTATE_180
            case 270:
                return cv2.ROTATE_90_CLOCKWISE
            case _:
                logging.warning(f"Tesseract gave unsupported angle: {angle}")
                return None

        return angle
    except Exception as e:
        logging.error(f"Tesseract OCR failed to detect angle: {e}")
        return None


def has_enough_text(pil_image: Image.Image, min_words=3) -> bool:
    """
    Return True if Tesseract detects at least `min_words` textual items in this image.
    Otherwise, likely not enough text to deskew safely.
    """

    try:
        data = pytesseract.image_to_data(pil_image, output_type=Output.DICT)
        words = [w.strip() for w in data["text"] if w.strip() != ""]
        return len(words) >= min_words
    except Exception as e:
        logging.error(f"{e}")
        return False


def debug_show_image(image: cv2.typing.MatLike, resize_image: bool = True):
    if resize_image:
        image = cv2.resize(image.copy(), (700, 1000))

    # We always use 'Debug Preview', so that the windows will
    # open up at the same position as the previously closed Window.
    cv2.imshow("Debug Preview", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def is_countour_a_line(contour: cv2.typing.MatLike) -> bool:
    _, (width, height), _ = cv2.minAreaRect(contour)
    return (width / height) >= 3 or (height / width) >= 3


# Rotate the image to correct the skew
def rotate_image(image: cv2.typing.MatLike, angle: float) -> cv2.typing.MatLike:
    (height, width, *_) = image.shape
    center = (width / 2, height / 2)

    # getRotationMatrix2D is created using the center of the image,
    # calculated angle, and scale factor of 1.0 (none)
    # warpAffine to apply the rotation matrix to the image
    # INTER_CUBIC uses cubic interpolation for smoother results and
    # to preserve img quality during rotation
    # BORDER_REPLICATE fills in empty areas after rotation by replicating
    # border pixels to prevent black corners or artifacts
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def deskew_image(
    pil_image: Image.Image,
    use_tesseract: bool = False,
    use_hough: bool = False,
    debug: bool = False,
) -> Image.Image:
    """
    Deskew input PIL.Image object with OpenCV.
    Return a new PIL.Image object with the correct skew.
    """

    # --------------------------------------------------
    # STEP 0: If not enough text, skip deskew.
    # --------------------------------------------------
    if use_tesseract and not has_enough_text(pil_image):
        logging.warning("Image has insufficient text for deskewing.")
        use_tesseract = False

    # --------------------------------------------------
    # 1. IMAGE PREP: format, grayscale, threshold
    # --------------------------------------------------

    # Convert PIL.Image to numpy array and RGB to BGR
    # OpenCV primarily uses BGR color space and PIL uses RGB
    # Convert for compatibility with OpenCV functions
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Use Tesseract OCR for angle detection.
    # Tesseract OCR is not capable of giving very miniscule angle detections,
    # but it's good at giving 90 degree intervals.
    # So, we will use it to ensure any text we find is correctly oriented.
    # Hough lines works significantly better if we perform this first.
    if use_tesseract and use_hough:
        tesseract_angle = get_skew_angle_tesseract(pil_image)
        if tesseract_angle is not None:
            cv2_image = cv2.rotate(cv2_image, tesseract_angle)
            if debug:
                debug_show_image(cv2_image)

    # Convert to grayscale to simplify the image to one color channel
    # Grayscale images are easier/faster to process, especially for
    # thresholding and contour detection, which rely on intensity values
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to binarize image to B&W from threshold value
    # calculated for smaller regions and in case of variable lighting
    # Separates text from background for effective contour detection
    # 255: maximum value assignable to a pixel
    # ADAPTIVE_THRESH_MEAN_C is mean of neighborhood area values minus constant
    # THRESH_BINARY_INV inverts the binary image (white text, black background)
    # 15: blockSize of a pixel neighborhood for calculating a threshold value
    # 10: constant value
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    # --------------------------------------------------
    # 2. IMAGE CONTOURS: morph, detect, filter, select
    # --------------------------------------------------

    # Use morphological operations, eliminate small artifacts/bridge line gaps
    # Define kernel, apply dilation => erosion to fill foreground objects holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Detect outlines in binarized image
    # RETR_EXTERNAL retrieves outermost contours for finding
    # main text block without internal noise
    # CHAIN_APPROX_SIMPLE compresses horizontal, vertical + diagonal segments,
    # reducing the number of points and simplifying contour representation
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour, typically the primary text block
    # The main text usually occupies the largest area
    # If no contours are found, return the original image unaltered
    if not contours:
        logging.warning("No contours found. Returning original image...")
        return pil_image

    # Draw all detected contours in red for debugging
    if debug:
        debug_image = cv2_image.copy()
        cv2.drawContours(debug_image, contours, -1, (0, 0, 255), 2)  # Red in BGR

    # -------------------------------------------------------------------------

    # Filter contours by area
    min_area = 100
    filtered_contours = [
        c
        for c in contours
        if cv2.contourArea(c) > min_area and not is_countour_a_line(c)
    ]

    if not filtered_contours:
        logging.warning("No large enough contours found. Returning original...")
        return pil_image

        # Draw all detected contours in red for debugging
    if debug:
        debug_image = cv2_image.copy()
        cv2.drawContours(debug_image, filtered_contours, -1, (255, 0, 0), 2)  # Blue BGR

    # Combine all filtered contours into one array of points
    all_contours = np.vstack(filtered_contours)

    # Compute the minimum area rectangle that encloses all contours
    encompassing_rect = cv2.minAreaRect(all_contours)
    box = cv2.boxPoints(encompassing_rect)
    box = np.int_(box)  # Convert to integer coordinates
    angle = encompassing_rect[-1]

    # Draw the encompassing bounding rectangle in green
    if debug:
        cv2.drawContours(debug_image, [box], 0, (0, 255, 0), 2)  # Green in RGB
        debug_show_image(debug_image)

    # -------------------------------------------------------------------------

    # --------------------------------------------------
    # 3. SKEW ANGLE: calculate, corroborate, correct
    # --------------------------------------------------

    # Determine rotation angle needed to correct the skew
    # minAreaRect computes smallest rectangle containing the contour;
    # provides both size and orientation
    # If angle < -45, the rectangle is more vertical than horizontal,
    # so adding 90 degrees adjusts it to represent the actual skew needed

    # Additional skew detection via Hough Line Transform enhances accuracy
    if use_hough:
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10,
        )

        # Draw the detected lines on the original image
        if debug and lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            debug_show_image(debug_image)

        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                current_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(current_angle)
            if angles:
                angle = np.median(angles)

    # Rotate the image to correct the skew
    deskewed = rotate_image(cv2_image, angle)

    if debug:
        debug_image = deskewed.copy()
        cv2.drawContours(debug_image, [box], 0, (0, 255, 0), 2)  # Green in BGR
        debug_show_image(debug_image)

    # Use Tesseract OCR for angle detection.
    # Tesseract OCR is not capable of giving very miniscule angle detections,
    # but it's good at giving 90 degree intervals.
    # So, we will use it to ensure any text we find is correctly oriented.
    if use_tesseract and not use_hough:
        tesseract_angle = get_skew_angle_tesseract(pil_image)
        if tesseract_angle is not None:
            deskewed = cv2.rotate(deskewed, tesseract_angle)
            if debug:
                debug_show_image(deskewed)

    # --------------------------------------------------
    # 4. FINAL PIL: convert, return
    # --------------------------------------------------

    # Convert from numpy array and return final image as PIL.Image object
    deskewed_pil = Image.fromarray(cv2.cvtColor(deskewed, cv2.COLOR_BGR2RGB))

    return deskewed_pil


def save_pil_images_as_pdf(
    pil_images: List[Image.Image],
    output_pdf_path: str,
    resize_factor: float = 1.0,
):
    """Save a list of PIL.Image objects as a multi-page PDF"""

    if not pil_images:
        logging.warning(f"No images to save for {output_pdf_path}.")
        return

    try:
        # Convert images to RGB
        pil_images = [p.convert("RGB") for p in pil_images]

        # Resize images to reduce PDF size
        # TODO: May remove this if it affects quality significantly
        if resize_factor != 1.0:
            pil_images = [
                p.resize(
                    (
                        int(p.width * resize_factor),
                        int(p.height * resize_factor),
                    ),
                    Image.ANTIALIAS,
                )
                for p in pil_images
            ]

        # Save first image and append the rest
        pil_images[0].save(output_pdf_path, save_all=True, append_images=pil_images[1:])
        logging.info(f"Saved PDF: {output_pdf_path}")
    except Exception as e:
        logging.error(f"Failed to save PDF {output_pdf_path}: {e}")


def convert_single_image(
    image_path: str,
    output_dir: str,
    debug: bool,
    use_tesseract: bool,
    use_hough: bool,
    resize_factor: float = 1.0,
):
    """
    Convert multi-frame image file to PDF.
    Allows for optional inversion and deskewing methods.
    """

    if not os.path.isfile(image_path):
        logging.error(f"File does not exist: {image_path}")
        return

    # Create output PDF path
    image_file_name = os.path.splitext(os.path.basename(image_path))[0]
    output_pdf_name = os.path.join(output_dir, f"{image_file_name}.pdf")

    # Read all image frames/pages
    pages = read_multiframe_image(image_path)
    if not pages:
        logging.warning(f"No pages found in {image_path}. Skipping...")
        return

    # Optionally log image info for debugging
    for page in pages:
        log_image_info(page)

    # Deskew each page
    deskewed_pages = []
    for page in pages:
        deskewed = deskew_image(page, use_tesseract, use_hough, debug=debug)
        deskewed_pages.append(deskewed)
        logging.info(f"Finished deskewing image {len(deskewed_pages)}/{len(pages)}")

    # Save deskewed pages as PDF
    save_pil_images_as_pdf(deskewed_pages, output_pdf_name, resize_factor)


def main():
    args = parse_arguments()
    setup_logging(args.log)

    if not os.path.isdir(args.output_dir):
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            logging.info(f"Created output directory: {args.output_dir}")
        except Exception as e:
            logging.error(f"Failed to create output directory {args.output_dir}: {e}")
            sys.exit(1)

    # Use multiprocessing for parallel processing
    with Pool(processes=min(len(args.image_paths), cpu_count())) as pool:
        args = [
            (
                path,
                args.log,
                args.output_dir,
                args.debug,
                args.use_hough,
                args.resize_factor,
            )
            for path in args.image_paths
        ]
        list(
            tqdm(
                pool.starmap(worker_process, args),
                total=len(args),
                desc="Converting multi-frame images",
            )
        )

    logging.info("Finished converting images! 🎉")


if __name__ == "__main__":
    main()
