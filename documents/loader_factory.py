"""
documents/loader_factory.py
Loader Factory tự động nhận diện định dạng tệp và khởi tạo Document Loader tương ứng.
Hỗ trợ đầy đủ các định dạng: .pdf, .docx, .txt, .csv, .md.
Chuẩn hóa dữ liệu đầu ra thành danh sách LangChain Document kèm Metadata chi tiết.
"""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from core.exceptions import DocumentLoadError, UnsupportedFileTypeError
from documents.hasher import DocumentHasher

logger = logging.getLogger("document_loader_factory")

# Khởi tạo hoặc tương thích với LangChain Document
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.docstore.document import Document  # type: ignore
    except ImportError:
        class Document:  # type: ignore
            """Fallback tương thích 100% với LangChain Document."""
            def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
                self.page_content = page_content
                self.metadata = metadata or {}

            def __repr__(self) -> str:
                return f"Document(page_content={self.page_content[:30]!r}..., metadata={self.metadata})"


class BaseFileParser:
    """Interface cơ sở cho các parser đọc tệp tin."""
    def parse(self, file_path: Path) -> List[Document]:
        raise NotImplementedError


class TextFileParser(BaseFileParser):
    """Parser xử lý tệp văn bản thuần (.txt)."""

    def parse(self, file_path: Path) -> List[Document]:
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "cp1258"]
        content = None
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            raise DocumentLoadError(
                str(file_path),
                "Không thể giải mã tệp văn bản với các bảng mã UTF-8/CP1258.",
            )

        return [
            Document(
                page_content=content,
                metadata={"source": str(file_path), "file_name": file_path.name},
            )
        ]


class MarkdownFileParser(BaseFileParser):
    """Parser xử lý tệp Markdown (.md)."""

    def parse(self, file_path: Path) -> List[Document]:
        text_parser = TextFileParser()
        docs = text_parser.parse(file_path)
        for doc in docs:
            doc.metadata["format"] = "markdown"
        return docs


class CSVFileParser(BaseFileParser):
    """Parser xử lý tệp dữ liệu bảng (.csv)."""

    def parse(self, file_path: Path) -> List[Document]:
        encodings = ["utf-8", "utf-8-sig", "cp1258", "latin-1"]
        content = None
        used_enc = "utf-8"
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                used_enc = enc
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise DocumentLoadError(str(file_path), "Không thể đọc nội dung tệp CSV.")

        docs: List[Document] = []
        reader = csv.DictReader(io.StringIO(content))
        for row_idx, row in enumerate(reader, start=1):
            row_text = ", ".join(f"{k}: {v}" for k, v in row.items() if v is not None)
            if row_text.strip():
                docs.append(
                    Document(
                        page_content=row_text,
                        metadata={
                            "source": str(file_path),
                            "file_name": file_path.name,
                            "row": row_idx,
                            "page": row_idx,
                        },
                    )
                )

        if not docs:
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": str(file_path), "file_name": file_path.name, "page": 1},
                )
            )

        return docs


class PDFFileParser(BaseFileParser):
    """Parser xử lý tệp tài liệu PDF (.pdf)."""

    def parse(self, file_path: Path) -> List[Document]:
        # Thử sử dụng LangChain PyPDFLoader
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(file_path))
            return loader.load()
        except ImportError:
            pass
        except Exception as e:
            logger.warning("PyPDFLoader thất bại: %s, chuyển sang pypdf native", e)

        # Fallback sang thư viện pypdf trực tiếp
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            docs: List[Document] = []
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(file_path),
                                "file_name": file_path.name,
                                "page": i,
                            },
                        )
                    )
            if docs:
                return docs
        except ImportError:
            pass
        except Exception as e:
            raise DocumentLoadError(str(file_path), f"Lỗi đọc nội dung PDF: {e}")

        # Fallback đọc binary / plain nếu chưa cài pypdf
        raise DocumentLoadError(
            str(file_path),
            "Vui lòng cài đặt thư viện 'pypdf' hoặc 'langchain-community' để đọc định dạng .pdf.",
        )


class DocxFileParser(BaseFileParser):
    """Parser xử lý tệp Microsoft Word (.docx)."""

    def parse(self, file_path: Path) -> List[Document]:
        # Thử sử dụng LangChain Docx2txtLoader
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(str(file_path))
            return loader.load()
        except ImportError:
            pass
        except Exception as e:
            logger.warning("Docx2txtLoader thất bại: %s, chuyển sang docx native", e)

        # Fallback sang python-docx
        try:
            import docx
            doc = docx.Document(str(file_path))
            full_text = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip():
                        full_text.append(row_text)
            content = "\n".join(full_text)
            return [
                Document(
                    page_content=content,
                    metadata={"source": str(file_path), "file_name": file_path.name, "page": 1},
                )
            ]
        except ImportError:
            pass
        except Exception as e:
            raise DocumentLoadError(str(file_path), f"Lỗi đọc nội dung .docx: {e}")

        # Fallback giải nén file XML bên trong docx (docx thực chất là file zip)
        try:
            import xml.etree.ElementTree as ET
            import zipfile
            with zipfile.ZipFile(str(file_path)) as zf:
                xml_content = zf.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            # Trích xuất toàn bộ text node trong namespace wordprocessingml
            texts = [node.text for node in tree.iter() if node.text and node.text.strip()]
            content = "\n".join(texts)
            return [
                Document(
                    page_content=content,
                    metadata={"source": str(file_path), "file_name": file_path.name, "page": 1},
                )
            ]
        except Exception as e:
            raise DocumentLoadError(
                str(file_path),
                f"Không thể đọc tệp docx. Vui lòng cài đặt thư viện 'python-docx' hoặc 'docx2txt': {e}",
            )


class DocumentLoaderFactory:
    """
    Factory tự động nhận diện và đọc nội dung tài liệu dựa trên phần mở rộng tệp.
    Hỗ trợ: .pdf, .docx, .txt, .csv, .md.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": PDFFileParser,
        ".docx": DocxFileParser,
        ".txt": TextFileParser,
        ".csv": CSVFileParser,
        ".md": MarkdownFileParser,
    }

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Trả về danh sách phần mở rộng tệp được hệ thống hỗ trợ."""
        return list(cls.SUPPORTED_EXTENSIONS.keys())

    @classmethod
    def is_supported(cls, file_path: Union[str, Path]) -> bool:
        """Kiểm tra tệp có nằm trong danh sách hỗ trợ không."""
        suffix = Path(file_path).suffix.lower()
        return suffix in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def load_document(cls, file_path: Union[str, Path]) -> List[Document]:
        """
        Nhận diện phần mở rộng, chọn Parser phù hợp và bóc tách nội dung tài liệu.
        Tự động bổ sung Metadata: source, file_name, file_type, file_size_bytes, file_hash.
        """
        path = Path(file_path)
        if not path.exists():
            raise DocumentLoadError(str(path), f"Tệp tin không tồn tại tại đường dẫn: {path}")

        if not path.is_file():
            raise DocumentLoadError(str(path), f"Đường dẫn không phải là một tệp hợp lệ: {path}")

        ext = path.suffix.lower()
        parser_cls = cls.SUPPORTED_EXTENSIONS.get(ext)
        if not parser_cls:
            raise UnsupportedFileTypeError(
                file_extension=ext,
                supported_types=cls.get_supported_extensions(),
            )

        try:
            # 1. Tính mã băm SHA-256 của tệp gốc
            file_hash = DocumentHasher.compute_file_sha256(path)
            file_size = path.stat().st_size

            # 2. Phân tích nội dung tệp
            parser = parser_cls()
            docs = parser.parse(path)

            if not docs:
                raise DocumentLoadError(str(path), "Tài liệu rỗng, không chứa nội dung văn bản.")

            # 3. Chuẩn hóa Metadata phong phú cho từng Document
            for idx, doc in enumerate(docs, start=1):
                doc.metadata.setdefault("source", str(path))
                doc.metadata["file_name"] = path.name
                doc.metadata["file_type"] = ext.lstrip(".")
                doc.metadata["file_size_bytes"] = file_size
                doc.metadata["file_hash"] = file_hash
                if "page" not in doc.metadata:
                    doc.metadata["page"] = idx

            return docs

        except (UnsupportedFileTypeError, DocumentLoadError):
            raise
        except Exception as e:
            logger.exception("Lỗi bất ngờ khi đọc tài liệu %s: %s", path, e)
            raise DocumentLoadError(str(path), str(e))
