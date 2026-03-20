# SYNAPSE - A Gateway of Intelligent Perception for Traffic Management
# Copyright (C) 2026 Noxfort Systems
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# File: src/utils/parquet_validator.py
# Author: Gabriel Moraes
# Date: 2026-02-27

import os
import logging
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

class ParquetValidator:
    """
    Utility class to validate .parquet datasets before ingestion.
    Ensures that the historical data meets the minimum requirements for SYNAPSE models.
    """

    REQUIRED_DAYS = 7

    @staticmethod
    def _detect_time_column(file_path: str) -> str:
        """
        Reads the parquet schema to auto-detect the temporal column.
        Avoids loading the entire dataframe to save memory.
        """
        schema = pq.read_schema(file_path)
        
        # Enhanced keywords inspired by robust analysis script
        keywords = ['data', 'date', 'time', 'dia', 'timestamp', 'ts']
        
        for col_name in schema.names:
            col_lower = col_name.lower()
            if any(kw in col_lower for kw in keywords):
                return col_name
                
        raise ValueError("Could not auto-detect a time column in the parquet schema.")

    @staticmethod
    def validate_minimum_timespan(file_path: str, time_column: str = None) -> bool:
        """
        Validates if the provided parquet file contains at least 7 unique calendar days of historical data.
        
        Args:
            file_path (str): The absolute or relative path to the .parquet dataset.
            time_column (str, optional): The name of the column containing timestamps. 
                                         If None, it will be auto-detected.
            
        Returns:
            bool: True if the dataset has data across at least 7 unique days, False otherwise.
            
        Raises:
            FileNotFoundError: If the provided parquet file path does not exist.
            ValueError: If the time column is missing or data is corrupted.
        """
        if not os.path.exists(file_path):
            logger.error(f"Dataset not found at path: {file_path}")
            raise FileNotFoundError(f"The parquet file {file_path} does not exist.")

        try:
            logger.info(f"Starting validation for dataset: {file_path}")
            
            # Auto-detect the time column if not explicitly provided
            if not time_column:
                time_column = ParquetValidator._detect_time_column(file_path)
                logger.info(f"Auto-detected time column: '{time_column}'")
            
            # Load only the specific time column to optimize memory usage
            df_time = pd.read_parquet(file_path, columns=[time_column])
            
            # Robust conversion: coerce errors to NaT and drop invalid rows
            df_time[time_column] = pd.to_datetime(df_time[time_column], errors='coerce', utc=True)
            df_time = df_time.dropna(subset=[time_column])
            
            if len(df_time) == 0:
                logger.warning("No valid data found after datetime conversion!")
                return False
            
            # Extract unique dates to ensure actual data spread based on calendar days
            df_time['unique_date'] = df_time[time_column].dt.date
            unique_days_count = df_time['unique_date'].nunique()
            
            min_date = df_time['unique_date'].min()
            max_date = df_time['unique_date'].max()
            
            logger.info(f"Dataset evaluated. Span: from {min_date} to {max_date}. Unique days with data: {unique_days_count}.")
            
            # Check if the unique calendar days count meets the requirement
            if unique_days_count >= ParquetValidator.REQUIRED_DAYS:
                logger.info("Validation passed: Dataset meets the minimum unique days requirement.")
                return True
            else:
                logger.warning(
                    f"Validation failed: Dataset has only {unique_days_count} unique days of data. "
                    f"Minimum required is {ParquetValidator.REQUIRED_DAYS} days."
                )
                return False
                
        except ValueError as ve:
            logger.error(f"Value error while parsing parquet file: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error validating parquet file: {str(e)}")
            raise