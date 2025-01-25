# Clone the repository

`git clone https://github.com/Coolcud/multiframe_image_to_pdf.git`

`cd multiframe_image_to_pdf`

# Create and activate a virtual environment

`python -m venv venv`

Windows: `venv\Scripts\activate`

macOS: `source venv/bin/activate`

# Install dependencies

`pip install -r requirements.txt`

# Install TesseractOCR

Download [here](https://github.com/tesseract-ocr/tesseract/releases)

# Install Latin TrainedData set for TesseractOCR

Download [here](https://github.com/tesseract-ocr/tessdata/blob/main/lat.traineddata)

# Execute program

`python pdfize.py <output_pdf_dir> <image_path1> [<image_path2> ...] [options]`

# Batch convert all tifs in folder (Windows/Batch Only)

```batch
SET INPUT_PATH=G:\Scans\TifFolder
for /f %f in ('dir /b "%INPUT_PATH%" ^| findstr /i .tif') do python pdfize.py "%INPUT_PATH%" "%INPUT_PATH%\%f"
```