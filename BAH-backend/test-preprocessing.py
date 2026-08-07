"""
Standalone test for app/core/preprocessing.py

Runs build_rgb_composite() against real sample TIR and WV files,
saves the result as a viewable PNG, and prints clear errors if anything breaks.

Usage (run from project root):
    python3 -m test_preprocessing
"""

from app.core.preprocessing import build_rgb_composite

# ---- Update these two paths to your actual sample files ----
TIR_FILE_PATH = "/Users/krish.sawant/Downloads/BAH-backend/samples/OR_ABI-L2-CMIPF-M6C13_G19_s20261690600228_e20261690609548_c20261690609597.nc"      # or .h5, whichever satellite you're testing
TIR_FILENAME = "OR_ABI-L2-CMIPF-M6C13_G19_s20261690600228_e20261690609548_c20261690609597.nc"               # must match extension, used for format detection

WV_FILE_PATH = "/Users/krish.sawant/Downloads/BAH-backend/samples/OR_ABI-L2-CMIPF-M6C08_G19_s20261690600228_e20261690609536_c20261690610007.nc"        # must be the SAME satellite as the TIR file
WV_FILENAME = "OR_ABI-L2-CMIPF-M6C08_G19_s20261690600228_e20261690609536_c20261690610007.nc"

OUTPUT_PATH = "/Users/krish.sawant/Downloads/BAH-backend/uploads/output2.png"


def main():
    print("Reading TIR file:", TIR_FILE_PATH)
    with open(TIR_FILE_PATH, "rb") as f:
        tir_bytes = f.read()

    print("Reading WV file:", WV_FILE_PATH)
    with open(WV_FILE_PATH, "rb") as f:
        wv_bytes = f.read()

    print("Running build_rgb_composite()...")
    try:
        png_bytes = build_rgb_composite(
            tir_bytes=tir_bytes,
            tir_filename=TIR_FILENAME,
            wv_bytes=wv_bytes,
            wv_filename=WV_FILENAME,
        )
    except Exception as e:
        print("FAILED — build_rgb_composite() raised an error:")
        print(type(e).__name__, "-", e)
        raise

    with open(OUTPUT_PATH, "wb") as f:
        f.write(png_bytes)

    print("SUCCESS")
    print(f"Composite image saved to: {OUTPUT_PATH}")
    print("Open it and check it looks like a reasonable image, not blank/garbage.")


if __name__ == "__main__":
    main()