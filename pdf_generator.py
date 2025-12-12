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
/* Use system fonts that WeasyPrint can access reliably */
body {
    font-family: 'DejaVu Sans', 'Arial', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif;
    line-height: 1.8;
    color: #1a1a1a;
    max-width: 100%;
    margin: 0;
    padding: 30px 40px;
    font-size: 11pt;
    background: white;
}

/* Main title - prominent and clear */
h1 {
    color: #000000;
    font-size: 26pt;
    font-weight: bold;
    margin-top: 0;
    margin-bottom: 30px;
    padding-bottom: 15px;
    border-bottom: 3px solid #333;
    line-height: 1.3;
}

/* Section headers - clear hierarchy */
h2 {
    color: #1a1a1a;
    font-size: 18pt;
    font-weight: bold;
    margin-top: 35px;
    margin-bottom: 15px;
    padding-bottom: 8px;
    border-bottom: 2px solid #666;
    line-height: 1.3;
}

/* Subsection headers */
h3 {
    color: #2a2a2a;
    font-size: 14pt;
    font-weight: bold;
    margin-top: 25px;
    margin-bottom: 12px;
    line-height: 1.3;
}

h4 {
    color: #3a3a3a;
    font-size: 12pt;
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 10px;
    line-height: 1.3;
}

/* Paragraphs - spacious and readable */
p {
    margin: 12px 0;
    text-align: left;
    line-height: 1.8;
}

/* Lists - better spacing */
ul, ol {
    margin: 15px 0;
    padding-left: 35px;
    line-height: 1.8;
}

li {
    margin: 8px 0;
}

/* Nested lists */
li > ul, li > ol {
    margin: 8px 0;
    padding-left: 25px;
}

/* Strong text */
strong {
    font-weight: bold;
    color: #000;
}

/* Emphasis */
em {
    font-style: italic;
    color: #2a2a2a;
}

/* Inline code */
code {
    background-color: #f5f5f5;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'DejaVu Sans Mono', 'Courier New', monospace;
    font-size: 10pt;
    color: #c7254e;
    border: 1px solid #e1e1e1;
}

/* Code blocks */
pre {
    background-color: #f8f8f8;
    padding: 15px;
    border-radius: 4px;
    border: 1px solid #ddd;
    border-left: 4px solid #666;
    font-size: 9.5pt;
    line-height: 1.6;
    margin: 20px 0;
}

pre code {
    background-color: transparent;
    padding: 0;
    border: none;
    color: #1a1a1a;
}

/* Blockquotes */
blockquote {
    border-left: 4px solid #ccc;
    padding-left: 20px;
    margin: 20px 0;
    color: #3a3a3a;
    font-style: italic;
}

/* Tables - clean and readable */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 10pt;
    border: 1px solid #ddd;
}

th {
    background-color: #333;
    color: white;
    padding: 12px 10px;
    text-align: left;
    font-weight: bold;
    border: 1px solid #222;
}

td {
    padding: 10px;
    border: 1px solid #ddd;
}

tr:nth-child(even) {
    background-color: #f9f9f9;
}

/* Horizontal rules - clear section breaks */
hr {
    border: none;
    border-top: 2px solid #ccc;
    margin: 40px 0;
}

/* Warning/Disclaimer boxes - highly visible */
/* Note: :contains() selector not supported in WeasyPrint */
/* Users should manually style disclaimer sections if needed */

/* Links - clear and readable */
a {
    color: #0066cc;
    text-decoration: underline;
}

/* Page break handling - keep logical sections together */
h1, h2, h3, h4 {
    page-break-after: avoid;
}

h2, h3, h4 {
    page-break-before: auto;
}

p, li {
    orphans: 3;
    widows: 3;
}

table, pre, blockquote {
    page-break-inside: avoid;
}

/* Page setup - proper margins and numbering */
@page {
    size: A4;
    margin: 2.5cm 2cm;

    @top-right {
        content: string(doctitle);
        font-size: 9pt;
        color: #666;
    }

    @bottom-center {
        content: "Page " counter(page);
        font-size: 9pt;
        color: #666;
    }
}

/* Specific styling for disclaimer sections */
p strong:first-child {
    display: block;
    margin-bottom: 10px;
    font-size: 12pt;
}

/* Better spacing around headers */
h1 + p, h2 + p, h3 + p {
    margin-top: 0;
}

/* References and cost sections */
/* Note: :contains() selector not supported in WeasyPrint */
/* These sections will use default styling */
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
