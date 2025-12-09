"""
Reference extractor - extracts references from LLM-generated text.
"""

import re
from typing import List
from ..models import ExtractedReference


class ReferenceExtractor:
    """Extracts references from text"""

    # Patterns for detecting references
    NUMBERED_REF_PATTERN = r'\[(\d+)\]'
    INLINE_CITATION_PATTERN = r'\(([A-Z][a-z]+\s+(?:et al\.\s+)?\d{4})\)'
    REFERENCE_SECTION_MARKERS = [
        '## References',
        '## Sources',
        '## Citations',
        'References:',
        'Sources:',
        'Citations:',
    ]

    def extract_from_text(self, text: str) -> List[ExtractedReference]:
        """
        Extract all references from text.

        Args:
            text: Text containing references

        Returns:
            List of ExtractedReference objects
        """
        references = []

        # Try to find dedicated reference section
        ref_section = self._extract_reference_section(text)

        if ref_section:
            # Parse reference section
            refs = self._parse_reference_section(ref_section)
            references.extend(refs)
        else:
            # Extract inline citations
            refs = self._extract_inline_citations(text)
            references.extend(refs)

        return references

    def _extract_reference_section(self, text: str) -> str:
        """Extract dedicated reference section if it exists"""
        for marker in self.REFERENCE_SECTION_MARKERS:
            if marker in text:
                parts = text.split(marker, 1)
                if len(parts) == 2:
                    return parts[1].strip()

        return ""

    def _parse_reference_section(self, ref_section: str) -> List[ExtractedReference]:
        """Parse a dedicated reference section"""
        references = []

        # Split by numbers [1], [2], etc. or by newlines
        # Try numbered format first
        parts = re.split(r'\n?\[(\d+)\]\s*', ref_section)

        if len(parts) > 2:
            # Numbered format found
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    ref_num = parts[i]
                    ref_text = parts[i + 1].strip()

                    if ref_text:
                        references.append(ExtractedReference(
                            raw_text=ref_text,
                            position=int(ref_num)
                        ))
        else:
            # Try line-by-line format
            lines = ref_section.split('\n')
            position = 1

            for line in lines:
                line = line.strip()
                if line and len(line) > 20:  # Reasonable minimum for a reference
                    # Remove leading markers like "- ", "* ", numbers, etc.
                    line = re.sub(r'^[-*•\d.)\]]\s*', '', line)

                    if line:
                        references.append(ExtractedReference(
                            raw_text=line,
                            position=position
                        ))
                        position += 1

        return references

    def _extract_inline_citations(self, text: str) -> List[ExtractedReference]:
        """Extract inline citations like (Author 2020) or [1]"""
        references = []

        # Find numbered citations [1]
        numbered_citations = re.findall(self.NUMBERED_REF_PATTERN, text)

        for num in set(numbered_citations):
            references.append(ExtractedReference(
                raw_text=f"[{num}]",
                position=int(num),
                citation_style="numbered"
            ))

        # Find inline citations (Author 2020)
        inline_citations = re.findall(self.INLINE_CITATION_PATTERN, text)

        for citation in set(inline_citations):
            references.append(ExtractedReference(
                raw_text=citation,
                citation_style="author-year"
            ))

        return references

    def extract_with_context(
        self,
        text: str,
        context_chars: int = 100
    ) -> List[ExtractedReference]:
        """
        Extract references with surrounding context.

        Args:
            text: Text containing references
            context_chars: Number of characters of context to include

        Returns:
            List of ExtractedReference with context populated
        """
        references = self.extract_from_text(text)

        # Add context for each reference
        for ref in references:
            # Find reference in text
            position = text.find(ref.raw_text)

            if position != -1:
                start = max(0, position - context_chars)
                end = min(len(text), position + len(ref.raw_text) + context_chars)

                context = text[start:end]
                ref.context = context

                # Try to extract the claim being supported
                claim = self._extract_claim_from_context(context, ref.raw_text)
                ref.claim = claim

        return references

    def _extract_claim_from_context(self, context: str, reference: str) -> str:
        """Extract the claim that the reference supports"""
        # Find the sentence containing the reference
        sentences = context.split('.')

        for sentence in sentences:
            if reference in sentence:
                # Clean up the sentence
                claim = sentence.replace(reference, '').strip()
                return claim

        return context[:200]  # Fallback to first 200 chars of context
