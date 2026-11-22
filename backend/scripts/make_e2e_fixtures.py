r"""Create the small PDFs used by the browser end-to-end tests (frontend/e2e/fixtures).

    cd backend
    ..\.venv\Scripts\python.exe scripts\make_e2e_fixtures.py
"""
from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parents[2] / "frontend" / "e2e" / "fixtures"

MANUAL_PAGES = [
    "Controller X200 - User Manual\nThis manual covers installation and maintenance of the X200 "
    "irrigation controller. The controller waters up to eight garden zones on a weekly schedule.",
    "Maintenance and Reset\nTo reset the controller to factory settings, hold the red RESET button for "
    "ten seconds until the display flashes. Replace the backup battery every two years. The replacement "
    "filter has part number ZX-4471 and fits all units made after 2019.",
    "Warranty\nThe X200 comes with a three year limited warranty covering defects in materials and "
    "workmanship. Water damage caused by incorrect installation is not covered.",
]


def write_pdf(path: Path, pages: list[str], **save_kwargs) -> None:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
    doc.save(path, **save_kwargs)
    doc.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_pdf(OUT / "X200_Manual.pdf", MANUAL_PAGES)
    write_pdf(
        OUT / "Garden_Pump.pdf",
        ["Garden Pump GP-9\nRinse the intake sieve every month. Spare sieve part number SV-118."],
    )
    write_pdf(
        OUT / "Encrypted.pdf",
        ["Secret contents"],
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-e2e",
        user_pw="user-e2e",
    )
    (OUT / "Corrupted.pdf").write_bytes(b"%PDF-1.7\nthis is not really a pdf\n%%EOF")
    (OUT / "setup.exe").write_bytes(b"MZ not a document")
    for f in sorted(OUT.iterdir()):
        print(f"{f.name}: {f.stat().st_size} bytes")


if __name__ == "__main__":
    main()
