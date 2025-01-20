# Clone the repository

`git clone https://github.com/Coolcud/multiframe_image_to_pdf.git`

`cd multiframe_image_to_pdf`

# Create and activate a virtual environment

`python -m venv venv`

Windows: `venv\Scripts\activate`

macOS: `source venv/bin/activate`

# Install dependencies

`pip install -r requirements.txt`

# Install Latin TrainedData set for TesseractOCR
Download [here](https://github.com/tesseract-ocr/tessdata/blob/main/lat.traineddata)

# Execute shell script

`./run_converter.sh <output_dir> <image_path1> [<image_path2> ...] [options]`

Example: `./run_converter.sh ./output ./test1.tif ./test2.tiff --invert --verbose --use-hough --resize 0.5`
