"""
Tabular Data Loader for CSV and Excel (.xlsx, .xls) files.
Extracts schema definitions, statistical summaries, and row-level structured records
to maximize semantic and numerical retrieval accuracy.
"""

from pathlib import Path
import pandas as pd
from app.ingestion.base_loader import BaseLoader, RawDocument
from app.utils.logger import logger


class TabularLoader(BaseLoader):
    """Loads CSV and Excel files into searchable structured passages with column analytics."""

    def load(self, file_path: Path) -> RawDocument:
        ext = file_path.suffix.lower()
        logger.info(f"Loading tabular document: {file_path.name} (format: {ext})")

        sections = []
        full_text_blocks = []
        total_rows = 0

        try:
            if ext == ".csv":
                df_dict = {"MainSheet": pd.read_csv(file_path)}
            elif ext in [".xlsx", ".xls"]:
                df_dict = pd.read_excel(file_path, sheet_name=None)
            else:
                raise ValueError(f"Unsupported tabular extension: {ext}")

            for sheet_name, df in df_dict.items():
                if df.empty:
                    continue

                df.columns = [str(c).strip() for c in df.columns]
                columns_list = list(df.columns)
                num_rows, num_cols = df.shape
                total_rows += num_rows

                # 1. Structural Header Summary
                sheet_summary_lines = [
                    f"### Sheet: {sheet_name}",
                    f"Columns: {', '.join(columns_list)}",
                    f"Total Records: {num_rows} rows x {num_cols} columns"
                ]

                # 2. Compute Aggregates & Descriptive Statistics for Numerical Columns
                num_cols_list = df.select_dtypes(include=["number"]).columns.tolist()
                if num_cols_list:
                    sheet_summary_lines.append("Numerical Column Statistics:")
                    for ncol in num_cols_list:
                        col_sum = df[ncol].sum()
                        col_mean = df[ncol].mean()
                        col_min = df[ncol].min()
                        col_max = df[ncol].max()
                        sheet_summary_lines.append(
                            f"  - {ncol}: Total={col_sum:,.2f}, Average={col_mean:,.2f}, Range=[{col_min:,.2f} to {col_max:,.2f}]"
                        )

                sheet_summary_text = "\n".join(sheet_summary_lines)
                full_text_blocks.append(sheet_summary_text)
                sections.append({
                    "sheet_name": sheet_name,
                    "is_summary": True,
                    "text": sheet_summary_text
                })

                # 3. Individual Record Representation
                for row_idx, row in df.iterrows():
                    fields = []
                    for col in columns_list:
                        val = row[col]
                        if pd.notna(val):
                            # Format numbers cleanly
                            if isinstance(val, (int, float)):
                                fields.append(f"{col}: {val}")
                            else:
                                fields.append(f"{col}: {val}")

                    if fields:
                        row_text = f"[{sheet_name} | Row {row_idx + 1}] " + " | ".join(fields)
                        full_text_blocks.append(row_text)
                        sections.append({
                            "sheet_name": sheet_name,
                            "row_index": row_idx + 1,
                            "text": row_text
                        })

            full_content = "\n\n".join(full_text_blocks)

            return RawDocument(
                content=full_content,
                source_filename=file_path.name,
                file_type="tabular",
                pages_or_sections=sections,
                metadata={
                    "format": ext.replace(".", ""),
                    "sheets": list(df_dict.keys()),
                    "total_rows": total_rows,
                    "file_size_bytes": file_path.stat().st_size
                }
            )

        except Exception as e:
            logger.error(f"Failed to parse tabular file {file_path.name}: {e}")
            raise ValueError(f"Could not read tabular file '{file_path.name}': {str(e)}")
