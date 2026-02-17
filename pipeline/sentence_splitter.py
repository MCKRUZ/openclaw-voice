"""Streaming sentence splitter for real-time TTS.

Buffers streaming text and yields complete sentences as soon as they're detected.
Optimized for low latency - starts TTS on first sentence while rest generates.
"""

import re
from dataclasses import dataclass
from typing import AsyncIterator, List

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Sentence:
    """A complete sentence ready for TTS."""

    text: str
    index: int  # Sentence number in stream (0-indexed)
    is_final: bool = False  # True if this is the last sentence


class StreamingSentenceSplitter:
    """
    Split streaming text into sentences in real-time.

    Detects sentence boundaries (. ! ? followed by space or newline)
    and yields complete sentences immediately for TTS processing.
    """

    # Sentence boundary patterns
    # Must have punctuation + whitespace or end of string
    SENTENCE_END_PATTERN = re.compile(
        r'([.!?])\s+|([.!?])$'
    )

    # Minimum sentence length to avoid fragmenting
    MIN_SENTENCE_LENGTH = 10

    def __init__(self):
        """Initialize sentence splitter."""
        self.buffer = ""
        self.sentence_count = 0

    def add_text(self, text: str) -> List[Sentence]:
        """
        Add streaming text chunk and extract complete sentences.

        Args:
            text: New text chunk from LLM stream

        Returns:
            List of complete sentences (may be empty if no boundaries found)
        """
        self.buffer += text
        return self._extract_sentences()

    def flush(self) -> List[Sentence]:
        """
        Flush remaining buffer as final sentence.

        Call this when stream is complete to get any remaining text.

        Returns:
            List containing final sentence (or empty if buffer is empty)
        """
        sentences = []

        if self.buffer.strip():
            sentence = Sentence(
                text=self.buffer.strip(),
                index=self.sentence_count,
                is_final=True,
            )
            sentences.append(sentence)
            self.sentence_count += 1
            logger.debug(
                f"Flushed final sentence #{sentence.index}: "
                f'"{sentence.text[:50]}..."'
            )

        self.buffer = ""
        return sentences

    def _extract_sentences(self) -> List[Sentence]:
        """
        Extract complete sentences from current buffer.

        Returns:
            List of complete sentences
        """
        sentences = []

        while True:
            # Find next sentence boundary
            match = self.SENTENCE_END_PATTERN.search(self.buffer)

            if not match:
                # No complete sentence yet
                break

            # Extract sentence up to boundary (including punctuation)
            end_pos = match.end()
            sentence_text = self.buffer[:end_pos].strip()

            # Check minimum length to avoid fragments
            if len(sentence_text) < self.MIN_SENTENCE_LENGTH:
                # Too short - might be abbreviation or fragment
                # Only break if we have more text coming, otherwise keep it
                if len(self.buffer) > end_pos + 10:
                    # More text after boundary - likely fragment, skip
                    self.buffer = self.buffer[end_pos:]
                    continue
                else:
                    # Close to end of buffer - keep as sentence
                    pass

            # Valid sentence found
            sentence = Sentence(
                text=sentence_text,
                index=self.sentence_count,
                is_final=False,
            )
            sentences.append(sentence)
            self.sentence_count += 1

            logger.debug(
                f"Extracted sentence #{sentence.index}: "
                f'"{sentence.text[:50]}..."'
            )

            # Remove sentence from buffer
            self.buffer = self.buffer[end_pos:].lstrip()

        return sentences

    def reset(self) -> None:
        """Reset splitter state for new stream."""
        self.buffer = ""
        self.sentence_count = 0


async def split_streaming_response(
    text_stream: AsyncIterator[str],
) -> AsyncIterator[Sentence]:
    """
    Split streaming LLM response into sentences in real-time.

    Args:
        text_stream: Async iterator yielding text chunks from LLM

    Yields:
        Complete sentences as they're detected
    """
    splitter = StreamingSentenceSplitter()

    try:
        async for chunk in text_stream:
            sentences = splitter.add_text(chunk)
            for sentence in sentences:
                yield sentence

        # Flush any remaining text as final sentence
        final_sentences = splitter.flush()
        for sentence in final_sentences:
            yield sentence

    except Exception as e:
        logger.error(f"Error in sentence splitting: {e}")
        # Flush buffer on error to avoid losing text
        final_sentences = splitter.flush()
        for sentence in final_sentences:
            yield sentence
        raise
