"""
Generic Document & Data Source Analytics Abstractions.
Supports multi-file, multi-format (XLSX, CSV, PDF) dataset modeling and resolution.
"""

import os
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DataSourceFormat(str, Enum):
    XLSX = "xlsx"
    XLS = "xls"
    CSV = "csv"
    PDF = "pdf"


class Dataset(BaseModel):
    """
    Represents an individual tabular dataset inside a data source
    (e.g., a sheet in an Excel file, a standalone CSV table, or an extracted table from a PDF).
    """
    dataset_id: str
    dataset_name: str
    sheet_name: Optional[str] = None
    page_range: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    data_types: Dict[str, str] = Field(default_factory=dict)
    numeric_measures: List[str] = Field(default_factory=list)
    text_dimensions: List[str] = Field(default_factory=list)
    date_fields: List[str] = Field(default_factory=list)
    row_count: int = 0
    storage_reference: str = ""  # SQLite table name or identifier


class DataSource(BaseModel):
    """
    Represents an uploaded document or analytical data source.
    """
    workspace_id: int
    source_id: str
    original_filename: str
    format: DataSourceFormat
    stored_path: str
    uploaded_at: str
    file_hash: str
    datasets: List[Dataset] = Field(default_factory=list)
    schema_summary: str = ""
    total_rows: int = 0
    db_path: Optional[str] = None


@dataclass
class DocumentAnalyticsContext:
    """
    Conversational active context for multi-turn document querying.
    """
    workspace_id: int
    source_id: Optional[str] = None
    source_name: Optional[str] = None
    source_format: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    active_source_id: Optional[str] = None
    active_dataset_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    previous_query_plan: Optional[dict] = None
    active_entity: Optional[str] = None
    active_metric: Optional[str] = None


class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"
    UNSUPPORTED_CROSS_SOURCE = "unsupported_cross_source"


class SourceResolutionResult(BaseModel):
    """
    Structured result of resolving a natural language query to data source(s) and dataset(s).
    """
    status: ResolutionStatus
    source_id: Optional[str] = None
    source: Optional[DataSource] = None
    dataset_ids: List[str] = Field(default_factory=list)
    selected_datasets: List[Dataset] = Field(default_factory=list)
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    reason: Optional[str] = None
    clarification_message: Optional[str] = None
    clarification_options: List[str] = Field(default_factory=list)
