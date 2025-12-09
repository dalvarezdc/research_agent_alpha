#!/usr/bin/env python3
"""
PDF Generator for Medical Analysis Reports
Converts markdown reports to beautifully formatted PDFs with emoji support.
"""

import logging
from pathlib import Path
from typing import Optional
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration


# Custom CSS for medical reports with emoji support
PDF_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

body {
    font-family: 'Inter', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif;
    line-height: 1.6;
    color: #2d3748;
    max-width: 800px;
    margin: 40px auto;
    padding: 20px;
    font-size: 11pt;
}

h1 {
    color: #1a202c;
    font-size: 28pt;
    font-weight: 700;
    margin-top: 0;
    margin-bottom: 20px;
    border-bottom: 3px solid #4299e1;
    padding-bottom: 10px;
}

h2 {
    color: #2d3748;
    font-size: 20pt;
    font-weight: 600;
    margin-top: 30px;
    margin-bottom: 15px;
    border-bottom: 2px solid #cbd5e0;
    padding-bottom: 8px;
}

h3 {
    color: #4a5568;
    font-size: 16pt;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 10px;
}

h4 {
    color: #718096;
    font-size: 14pt;
    font-weight: 600;
    margin-top: 15px;
    margin-bottom: 8px;
}

p {
    margin: 10px 0;
    text-align: justify;
}

ul, ol {
    margin: 10px 0;
    padding-left: 30px;
}

li {
    margin: 5px 0;
}

strong {
    font-weight: 600;
    color: #1a202c;
}

em {
    font-style: italic;
    color: #4a5568;
}

code {
    background-color: #f7fafc;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 10pt;
    color: #e53e3e;
}

pre {
    background-color: #f7fafc;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #4299e1;
    overflow-x: auto;
    font-size: 9pt;
}

pre code {
    background-color: transparent;
    padding: 0;
    color: #2d3748;
}

blockquote {
    border-left: 4px solid #e2e8f0;
    padding-left: 20px;
    margin: 15px 0;
    color: #4a5568;
    font-style: italic;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    font-size: 10pt;
}

th {
    background-color: #4299e1;
    color: white;
    padding: 10px;
    text-align: left;
    font-weight: 600;
}

td {
    padding: 8px 10px;
    border-bottom: 1px solid #e2e8f0;
}

tr:nth-child(even) {
    background-color: #f7fafc;
}

hr {
    border: none;
    border-top: 2px solid #e2e8f0;
    margin: 30px 0;
}

/* Emoji styling */
.emoji {
    font-size: 1.2em;
}

/* Warning/disclaimer boxes */
p:has(> strong:first-child) {
    background-color: #fff5f5;
    border-left: 4px solid #fc8181;
    padding: 15px;
    margin: 20px 0;
    border-radius: 4px;
}

/* Footer styling */
footer {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 2px solid #e2e8f0;
    font-size: 9pt;
    color: #718096;
    text-align: center;
}

/* Page break handling */
h1, h2, h3 {
    page-break-after: avoid;
}

table, pre, blockquote {
    page-break-inside: avoid;
}

/* Links */
a {
    color: #4299e1;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Print-specific styles */
@page {
    size: A4;
    margin: 2cm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9pt;
        color: #718096;
    }
}
"""


def markdown_to_pdf(
    markdown_file: str,
    pdf_file: Optional[str] = None,
    custom_css: Optional[str] = None
) -> str:
    """
    Convert a markdown file to a beautifully formatted PDF.

    Args:
        markdown_file: Path to the input markdown file
        pdf_file: Path to the output PDF file (auto-generated if None)
        custom_css: Additional CSS to apply (optional)

    Returns:
        Path to the generated PDF file

    Raises:
        FileNotFoundError: If markdown file doesn't exist
        Exception: If PDF generation fails
    """
    logger = logging.getLogger(__name__)

    # Validate input file
    md_path = Path(markdown_file)
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_file}")

    # Generate output filename if not provided
    if pdf_file is None:
        pdf_file = str(md_path.with_suffix('.pdf'))

    pdf_path = Path(pdf_file)

    try:
        # Read markdown content
        logger.info(f"Reading markdown file: {markdown_file}")
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # Convert markdown to HTML with extensions
        logger.info("Converting markdown to HTML")
        html_content = markdown.markdown(
            md_content,
            extensions=[
                'extra',  # Tables, fenced code blocks, etc.
                'codehilite',  # Syntax highlighting
                'toc',  # Table of contents
                'nl2br',  # New line to break
                'sane_lists',  # Better list handling
            ],
            extension_configs={
                'codehilite': {
                    'css_class': 'highlight',
                    'linenums': False,
                }
            }
        )

        # Wrap HTML with proper structure
        full_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Medical Analysis Report</title>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # Prepare CSS
        css_list = [CSS(string=PDF_CSS)]
        if custom_css:
            css_list.append(CSS(string=custom_css))

        # Generate PDF
        logger.info(f"Generating PDF: {pdf_file}")
        font_config = FontConfiguration()

        html_obj = HTML(string=full_html)
        html_obj.write_pdf(
            pdf_path,
            stylesheets=css_list,
            font_config=font_config
        )

        logger.info(f"✓ PDF generated successfully: {pdf_file}")
        return str(pdf_path)

    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}")
        raise Exception(f"PDF generation failed: {e}") from e


def convert_markdown_to_pdf_safe(
    markdown_file: str,
    pdf_file: Optional[str] = None
) -> Optional[str]:
    """
    Safely convert markdown to PDF with error handling.

    Args:
        markdown_file: Path to the input markdown file
        pdf_file: Path to the output PDF file (optional)

    Returns:
        Path to generated PDF, or None if failed
    """
    logger = logging.getLogger(__name__)

    try:
        return markdown_to_pdf(markdown_file, pdf_file)
    except Exception as e:
        logger.warning(f"PDF generation skipped: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_generator.py <markdown_file> [pdf_file]")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    md_file = sys.argv[1]
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        output = markdown_to_pdf(md_file, pdf_file)
        print(f"✓ PDF generated: {output}")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
