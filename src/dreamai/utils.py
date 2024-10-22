import inspect
import quopri
import re
import textwrap
import traceback
from itertools import chain
from pathlib import Path
from typing import Final, Iterable

import demoji

from dreamai.settings import RAGSettings

rag_settings = RAGSettings()

CHUNK_SIZE = rag_settings.chunk_size
CHUNK_OVERLAP = rag_settings.chunk_overlap
MIN_CHUNK_SIZE = rag_settings.min_chunk_size
SEPARATORS = rag_settings.separators
UNICODE_BULLETS: Final[list[str]] = [
    "\u0095",
    "\u2022",
    "\u2023",
    "\u2043",
    "\u3164",
    "\u204c",
    "\u204d",
    "\u2219",
    "\u25cb",
    "\u25cf",
    "\u25d8",
    "\u25e6",
    "\u2619",
    "\u2765",
    "\u2767",
    "\u29be",
    "\u29bf",
    "\u002d",
    "",
    "\x95",
    "·",
]
BULLETS_PATTERN = "|".join(UNICODE_BULLETS)
UNICODE_BULLETS_RE_0W = re.compile(f"(?={BULLETS_PATTERN})(?<!{BULLETS_PATTERN})")
UNICODE_BULLETS_RE = re.compile(f"(?:{BULLETS_PATTERN})(?!{BULLETS_PATTERN})")
E_BULLET_PATTERN = re.compile(r"^e(?=\s)", re.MULTILINE)
PARAGRAPH_PATTERN = r"\s*\n\s*"
PARAGRAPH_PATTERN_RE = re.compile(
    f"((?:{BULLETS_PATTERN})|{PARAGRAPH_PATTERN})(?!{BULLETS_PATTERN}|$)",
)
DOUBLE_PARAGRAPH_PATTERN_RE = re.compile("(" + PARAGRAPH_PATTERN + "){2}")


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE,
    keep_separator: bool = True,
    separators: list[str] | None = None,
) -> list[dict]:
    if chunk_size == 0 or len(text) <= chunk_size:
        return [{"text": text, "start": 0, "end": len(text)}]
    chunk_overlap = min(chunk_overlap, chunk_size // 2)
    separators = separators or [r"#{1,6}\s+.+", r"\*\*.*?\*\*", r"---", r"\n\n", r"\n"]
    pattern = f'({"|".join(separators)})' if keep_separator else f'(?:{"|".join(separators)})'
    chunks = [chunk.strip() for chunk in re.split(pattern, text) if chunk.strip()]
    result = []
    current_chunk = ""
    start_index = 0
    for chunk in chunks:
        if (
            not current_chunk
            or len(current_chunk) + len(chunk) <= chunk_size
            or (
                (len(chunk) < min_chunk_size)
                and (len(current_chunk) + len(chunk) <= chunk_size * 2)
            )
        ):
            current_chunk += (" " if current_chunk else "") + chunk
        else:
            if current_chunk:
                end_index = start_index + len(current_chunk)
                result.append(
                    {
                        "text": current_chunk,
                        "start": max(0, start_index),
                        "end": end_index,
                    }
                )
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    chunk = current_chunk[-chunk_overlap:] + " " + chunk
                    start_index = end_index - chunk_overlap
                else:
                    start_index = end_index
            current_chunk = chunk
    if current_chunk:
        result.append(
            {
                "text": current_chunk,
                "start": start_index,
                "end": start_index + len(current_chunk),
            }
        )
    return result


def run_code(code, *args, **kwargs):
    try:
        function_def = "def " + code.split("def ")[1].split("\n")[0]
        code_before, code_after = code.split(function_def)
        code = f"{function_def}\n{' '*4}{code_before.strip()}\n{code_after}"
        func = code.split("def ")[1].split("(")[0]
        exec(code)
        func = locals()[func]
        return func(*args, **kwargs)
    except Exception:
        return traceback.format_exc()


def flatten(o: Iterable):
    for item in o:
        if isinstance(item, str):
            yield item
            continue
        try:
            yield from flatten(item)
        except TypeError:
            yield item


def to_camel(s: str, sep: str = "_") -> str:
    if sep not in s:
        return s
    return "".join(s.title().split(sep))


def to_snake(s: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def noop(x=None, *args, **kwargs):  # noqa
    return x


def resolve_data_path(
    data_path: list[str | Path] | str | Path, file_extension: str | None = None
) -> chain:
    if not isinstance(data_path, list):
        data_path = [data_path]
    paths = []
    for dp in flatten(data_path):
        if isinstance(dp, (str, Path)):
            dp = Path(dp)
            if not dp.exists():
                raise Exception(f"Path {dp} does not exist.")
            if dp.is_dir():
                if file_extension:
                    paths.append(dp.glob(f"*.{file_extension}"))
                else:
                    paths.append(dp.iterdir())
            else:
                if file_extension is None or dp.suffix == f".{file_extension}":
                    paths.append([dp])
    return chain(*paths)


def flatten_list(my_list: list) -> list:
    new_list = []
    for x in my_list:
        if isinstance(x, list):
            new_list += flatten_list(x)
        else:
            new_list.append(x)
    return new_list


def deindent(text: str) -> str:
    return textwrap.dedent(inspect.cleandoc(text))


def remove_digits(text: str) -> str:
    return re.sub(r"\d+", "", text)


def dict_to_xml(d: dict) -> str:
    xml_str = ""
    for key, value in d.items():
        xml_str += f"<{key}>\n{value}\n</{key}>\n"
    return xml_str.strip()


def insert_xml_tag(text: str, tag: str, start: int, end: int) -> str:
    if not tag.startswith("<"):
        tag = f"<{tag}>"
    return text[:start] + tag + text[start:end] + f"</{tag[1:]}" + text[end:]


def _txt_to_content(content_file: str | Path) -> str:
    return deindent(Path(content_file).read_text()) if Path(content_file).exists() else ""


def _process_content(content: str | Path | list[str]) -> str:
    if not content:
        return ""
    if isinstance(content, list):
        content = "\n\n---\n\n".join(list(content))
    elif isinstance(content, (Path, str)):
        content = str(content)
        if content.endswith(".txt"):
            content = _txt_to_content(content)
    return deindent(str(content))


def format_encoding_str(encoding: str) -> str:
    formatted_encoding = encoding.lower().replace("_", "-")
    annotated_encodings = ["iso-8859-6-i", "iso-8859-6-e", "iso-8859-8-i", "iso-8859-8-e"]
    if formatted_encoding in annotated_encodings:
        formatted_encoding = formatted_encoding[:-2]
    return formatted_encoding


def bytes_string_to_string(text: str, encoding: str = "utf-8"):
    return bytes([ord(char) for char in text]).decode(format_encoding_str(encoding))


def clean_non_ascii_chars(text) -> str:
    return text.encode("ascii", "ignore").decode()


def group_bullet_paragraph(paragraph: str) -> list:
    paragraph = (re.sub(E_BULLET_PATTERN, "·", paragraph)).strip()
    bullet_paras = re.split(UNICODE_BULLETS_RE_0W, paragraph)
    return [re.sub(PARAGRAPH_PATTERN, " ", bullet) for bullet in bullet_paras if bullet]


def group_broken_paragraphs(
    text: str,
    line_split: re.Pattern[str] = PARAGRAPH_PATTERN_RE,
    paragraph_split: re.Pattern[str] = DOUBLE_PARAGRAPH_PATTERN_RE,
) -> str:
    paragraphs = paragraph_split.split(text)
    clean_paragraphs = []
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        para_split = line_split.split(paragraph)
        all_lines_short = all(len(line.strip().split(" ")) < 5 for line in para_split)
        if UNICODE_BULLETS_RE.match(paragraph.strip()) or E_BULLET_PATTERN.match(
            paragraph.strip()
        ):
            clean_paragraphs.extend(group_bullet_paragraph(paragraph))
        elif all_lines_short:
            clean_paragraphs.extend([line for line in para_split if line.strip()])
        else:
            clean_paragraphs.append(re.sub(PARAGRAPH_PATTERN, " ", paragraph))
    return "\n\n".join(clean_paragraphs)


def replace_mime_encodings(text: str, encoding: str = "utf-8") -> str:
    formatted_encoding = format_encoding_str(encoding)
    return quopri.decodestring(text.encode(formatted_encoding)).decode(formatted_encoding)


def replace_unicode_quotes(text: str) -> str:
    text = text.replace("\x91", "‘")
    text = text.replace("\x92", "’")
    text = text.replace("\x93", "“")
    text = text.replace("\x94", "”")
    text = text.replace("&apos;", "'")
    text = text.replace("â\x80\x99", "'")
    text = text.replace("â\x80“", "—")
    text = text.replace("â\x80”", "–")
    text = text.replace("â\x80˜", "‘")
    text = text.replace("â\x80¦", "…")
    text = text.replace("â\x80™", "’")
    text = text.replace("â\x80œ", "“")
    text = text.replace("â\x80?", "”")
    text = text.replace("â\x80ť", "”")
    text = text.replace("â\x80ś", "“")
    text = text.replace("â\x80¨", "—")
    text = text.replace("â\x80ł", "″")
    text = text.replace("â\x80Ž", "")
    text = text.replace("â\x80‚", "")
    text = text.replace("â\x80‰", "")
    text = text.replace("â\x80‹", "")
    text = text.replace("â\x80", "")
    text = text.replace("â\x80s'", "")
    return text


def clean_text(
    text: str, no_digits: bool = False, no_emojis: bool = False, group: bool = False
) -> str:
    text = re.sub(r"[\n+]", "\n", text)
    text = re.sub(r"[\t+]", " ", text)
    text = re.sub(r"[. .]", " ", text)
    text = re.sub(r"([ ]{2,})", " ", text)
    try:
        text = bytes_string_to_string(text)
    except Exception:
        pass
    try:
        text = clean_non_ascii_chars(text)
    except Exception:
        pass
    try:
        text = replace_unicode_quotes(text)
    except Exception:
        pass
    try:
        text = replace_mime_encodings(text)
    except Exception:
        pass
    if group:
        try:
            text = "\n".join(group_bullet_paragraph(text))
        except Exception:
            pass
        try:
            text = group_broken_paragraphs(text)
        except Exception:
            pass
    if no_digits:
        text = remove_digits(text)
    if no_emojis:
        text = demoji.replace(text, "")
    return text.strip()
