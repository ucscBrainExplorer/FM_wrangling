#!/usr/bin/env python3
"""
extract_pdf_glossary.py — extract a cell type glossary from a paper PDF.

Reads a PDF, extracts text in batches of pages, and asks an LLM to produce a
compact "LABEL: plain English name" glossary. Subsequent batches only add entries
not already found in earlier ones. Output is written to <pdf_stem>_glossary.txt
(or --output). If the output file already exists it is loaded and printed without
re-running the LLM (use --force to override).

This glossary file is consumed by generate_remap.py via --pdf.

Usage
-----
  python3 extract_pdf_glossary.py --pdf paper.pdf
  python3 extract_pdf_glossary.py --pdf paper.pdf --api gemini
  python3 extract_pdf_glossary.py --pdf paper.pdf --output custom_glossary.txt --force

Environment
-----------
  ANTHROPIC_API_KEY   required for --api claude
  GOOGLE_API_KEY      required for --api gemini
  (default: gemini if GOOGLE_API_KEY is set, else claude)
"""

import argparse
import os
import re
import sys
import time


def pdf_page_count(pdf_path):
    try:
        import pypdf
        return len(pypdf.PdfReader(pdf_path).pages)
    except Exception as e:
        print(f"Error reading PDF: {e}", file=sys.stderr)
        return 0


def extract_pdf_text(pdf_path, max_pages=20, start_page=0):
    """Extract plain text from a PDF, capped at max_pages pages."""
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        pages = reader.pages[start_page:start_page + max_pages]
        text = '\n'.join(page.extract_text() or '' for page in pages)
        return text.strip() or None
    except Exception as e:
        print(f"  Warning: could not extract text: {e}", file=sys.stderr)
        return None


def extract_cell_type_glossary(pdf_text, api='claude'):
    """Ask an LLM to extract a compact cell type glossary from paper text.

    Returns a string of "LABEL: plain English name" lines, or None on failure.
    """
    from annotate_cl_terms import _load_prompt

    words = pdf_text.split()
    if len(words) > 6000:
        pdf_text = ' '.join(words[:6000]) + ' [truncated]'

    prompt = _load_prompt("glossary_extraction.txt", pdf_text=pdf_text)

    def _call():
        if api == 'claude':
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        elif api == 'gemini':
            from google import genai as google_genai
            client = google_genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
            response = client.models.generate_content(
                model='gemini-2.5-flash', contents=prompt)
            return ''.join(
                p.text for p in response.candidates[0].content.parts
                if hasattr(p, 'text')
            ).strip()
        return None

    for attempt in range(3):
        try:
            return _call()
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ('credit', 'billing', 'unauthorized',
                                       'invalid_api_key', 'permission')):
                print(f"  Fatal API error: {e}", file=sys.stderr)
                return None
            if attempt < 2 and any(k in err for k in ('529', 'overloaded',
                                                        '429', 'rate_limit', 'timeout')):
                wait = 10 * (attempt + 1)
                print(f"  {api} overloaded — retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"  Glossary extraction failed: {e}", file=sys.stderr)
            return None
    return None


def _parse_keys(glossary_text):
    """Return set of normalized keys already in glossary text."""
    keys = set()
    for line in glossary_text.splitlines():
        sep = ':' if ':' in line else ('=' if '=' in line else None)
        if sep:
            key, _, _ = line.partition(sep)
            key = re.sub(r'[–—‒]', '-', key.strip().lower())
            if key:
                keys.add(key)
    return keys


def main():
    parser = argparse.ArgumentParser(
        description="Extract a cell type glossary from a paper PDF"
    )
    parser.add_argument("--pdf", required=True,
                        help="Path to the source paper PDF")
    parser.add_argument("--api", choices=["claude", "gemini"], default=None,
                        help="LLM API to use (default: gemini if key set, else claude)")
    parser.add_argument("--output", default=None,
                        help="Output path (default: <pdf_stem>_glossary.txt)")
    parser.add_argument("--pages", type=int, default=20,
                        help="Pages per batch (default: 20)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if cached glossary file already exists")
    args = parser.parse_args()

    if args.output is None:
        args.output = re.sub(r'\.pdf$', '', args.pdf, flags=re.IGNORECASE) + '_glossary.txt'

    has_anthropic = bool(os.environ.get('ANTHROPIC_API_KEY'))
    has_gemini    = bool(os.environ.get('GOOGLE_API_KEY'))
    if args.api is None:
        args.api = 'gemini' if has_gemini else ('claude' if has_anthropic else None)
    if args.api is None:
        print("Error: set ANTHROPIC_API_KEY or GOOGLE_API_KEY", file=sys.stderr)
        sys.exit(1)
    if args.api == 'claude' and not has_anthropic:
        print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    if args.api == 'gemini' and not has_gemini:
        print("Error: GOOGLE_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Load cached glossary if present and --force not set
    if not args.force and os.path.exists(args.output):
        with open(args.output) as f:
            glossary = f.read().strip()
        entries = [l for l in glossary.splitlines() if l.strip()]
        print(f"Loaded cached glossary from {args.output} ({len(entries)} entries):")
        for e in entries:
            print(f"  {e}")
        return

    total_pages = pdf_page_count(args.pdf)
    if total_pages == 0:
        sys.exit(1)
    print(f"PDF: {args.pdf} ({total_pages} pages, batch size: {args.pages})")

    # First batch
    end = min(args.pages, total_pages)
    print(f"\nExtracting pages 1–{end} (API: {args.api})...")
    text = extract_pdf_text(args.pdf, max_pages=args.pages, start_page=0)
    if not text:
        print("Error: could not extract text from PDF", file=sys.stderr)
        sys.exit(1)

    glossary = extract_cell_type_glossary(text, api=args.api)
    if not glossary:
        print("Error: LLM returned no glossary", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(glossary.splitlines())} entries")

    # Additional batches — only add entries with new keys
    page = args.pages
    while page < total_pages:
        end = min(page + args.pages, total_pages)
        print(f"\nPages {page + 1}–{end} of {total_pages}...")
        extra_text = extract_pdf_text(args.pdf, max_pages=args.pages, start_page=page)
        if extra_text:
            supplemental = extract_cell_type_glossary(extra_text, api=args.api)
            if supplemental:
                existing_keys = _parse_keys(glossary)
                new_lines = []
                for line in supplemental.splitlines():
                    sep = ':' if ':' in line else ('=' if '=' in line else None)
                    if sep:
                        key, _, _ = line.partition(sep)
                        key = re.sub(r'[–—‒]', '-', key.strip().lower())
                        if key and key not in existing_keys:
                            new_lines.append(line)
                            existing_keys.add(key)
                if new_lines:
                    glossary += '\n' + '\n'.join(new_lines)
                    print(f"  +{len(new_lines)} new entries")
                else:
                    print("  no new entries")
        page += args.pages

    with open(args.output, 'w') as f:
        f.write(glossary + '\n')

    entries = [l for l in glossary.splitlines() if l.strip()]
    print(f"\nWritten {len(entries)} entries to {args.output}")
    for e in entries:
        print(f"  {e}")


if __name__ == "__main__":
    main()
