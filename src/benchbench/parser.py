from pydantic import BaseModel, ValidationError
from typing import Generator
from enum import StrEnum
from pathlib import Path
from yaml import safe_load
import logging

MD_HEADER_PREFIX = "# "

logger = logging.getLogger(__name__)


class Roles(StrEnum):
    system = "system"
    assistant = "assistant"
    user = "user"


class Message(BaseModel):
    role: Roles
    content: str


class Content(BaseModel):
    messages: list[Message]


class Frontmatter(BaseModel):
    id: str


class MDConfig(BaseModel):
    frontmatter: Frontmatter
    content: Content | None


def walk_md_content(content: str) -> Generator[Message, None, None]:
    lines = content.split("\n")
    role: Roles | None = None
    block_lines = []
    inside_valid_block = False
    for line in lines:
        logger.debug(f"{line=}")
        if line.startswith(MD_HEADER_PREFIX):
            # yield previous block if exists
            if inside_valid_block:
                assert role is not None, "Role can't be None if valid_block ended"
                yield Message(role=role, content="\n".join(block_lines).strip())
                block_lines = []
            header = line[len(MD_HEADER_PREFIX) :].lower()
            try:
                role = Roles(header)
                inside_valid_block = True
            except ValueError:
                raise ValueError(
                    f"Unspported Markdown Header. Expected one of 'System', 'Assistant', 'User'. Got: {header}"
                )
            continue
        if inside_valid_block:
            block_lines.append(line)
    if inside_valid_block:
        assert role is not None, "Role can't be None if valid_block ended"
        yield Message(role=role, content="\n".join(block_lines).strip())


def parse_md(path: Path) -> MDConfig | None:
    if not path.exists():
        logger.error(f"Markdown file with path {path} doesn't exist")
        return None

    with open(path, "r") as file:
        text = file.read()

    # Extract frontmatter
    frontmatter_pos_start = text.find("---\n")
    frontmatter_pos_end = text.find("---\n", frontmatter_pos_start + 4)
    frontmatter_content = text[frontmatter_pos_start + 4 : frontmatter_pos_end].strip()
    obj = safe_load(frontmatter_content)

    try:
        frontmatter = Frontmatter.model_validate(obj)
    except ValidationError:
        logger.error(f"Unable to validate frontmatter object in file: {str(path)}")
        return None

    # After frontmatter we have md text content
    text_content = text[frontmatter_pos_end + 4 :]
    messages = list(walk_md_content(text_content))
    content = Content(messages=messages) if messages else None

    return MDConfig(frontmatter=frontmatter, content=content)
