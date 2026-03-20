# SYNAPSE - A Gateway of Intelligent Perception for Traffic Management
# Copyright (C) 2025 Noxfort Systems
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
# File: ui/wizards/import_wizard.py
# Author: Gabriel Moraes
# Date: 2025-11-30

import sqlite3
import pandas as pd
import os
from PyQt6.QtCore import Qt, pyqtSlot, QThread
from PyQt6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QLabel, QLineEdit, 
    QPushButton, QFileDialog, QFormLayout, QSpinBox, QListWidget,
    QProgressBar, QTextEdit, QMessageBox
)
from ui.styles.theme_manager import ThemeManager

from src.services.database_importer import DatabaseImporter

class ImportWizard(QWizard):
    """
    Wizard dialog to guide the user through the 'Raw Data -> Base DB' process.
    Supports both Parquet (New Standard) and SQLite (Legacy).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Import Historical Data"))
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.resize(600, 450)
        
        # Shared State
        self.source_path = ""
        self.target_freq = 1.0
        
        # Pages
        self.page_intro = IntroPage(self)
        self.page_config = ConfigPage(self)
        self.page_process = ProcessingPage(self)
        
        self.addPage(self.page_intro)
        self.addPage(self.page_config)
        self.addPage(self.page_process)

class IntroPage(QWizardPage):
    def __init__(self, wizard):
        super().__init__(wizard)
        self.setTitle(self.tr("Select Source Data"))
        self.setSubTitle(self.tr("Choose the raw data file (.parquet or .db) containing historical traffic."))
        
        layout = QVBoxLayout(self)
        
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(self.tr("Path to .parquet or .db file..."))
        self.path_edit.setReadOnly(True)
        
        btn_browse = QPushButton(self.tr("Browse..."))
        btn_browse.clicked.connect(self._browse)
        
        form = QFormLayout()
        form.addRow(self.tr("Source File:"), self.path_edit)
        form.addRow("", btn_browse)
        
        layout.addLayout(form)
        
        # Validation
        self.registerField("source_path*", self.path_edit) # * means mandatory

    def _browse(self):
        # Updated filter to prioritize Parquet
        f, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Data Source", 
            "", 
            "All Supported (*.parquet *.db);;Parquet Files (*.parquet);;SQLite Database (*.db)"
        )
        if f:
            self.path_edit.setText(f)
            self.wizard().source_path = f

class ConfigPage(QWizardPage):
    def __init__(self, wizard):
        super().__init__(wizard)
        self.setTitle(self.tr("Configuration & Inspection"))
        self.setSubTitle(self.tr("Review found sources and define the target timebase."))
        
        layout = QVBoxLayout(self)
        
        # Inspection List
        layout.addWidget(QLabel(self.tr("Sources found in file:")))
        self.list_tables = QListWidget()
        layout.addWidget(self.list_tables)
        
        # Frequency Setting
        self.spin_freq = QSpinBox()
        self.spin_freq.setRange(1, 60)
        self.spin_freq.setValue(1)
        self.spin_freq.setSuffix(self.tr(" min"))
        self.spin_freq.setToolTip(self.tr("The common interval to synchronize all sensors."))
        
        form = QFormLayout()
        form.addRow(self.tr("Target Interval (Base):"), self.spin_freq)
        layout.addLayout(form)
        
        self.registerField("target_freq", self.spin_freq)

    def initializePage(self):
        """Called when user enters this page. We scan the file here."""
        path = self.wizard().source_path
        self.list_tables.clear()
        
        if not path or not os.path.exists(path):
            self.list_tables.addItem(self.tr("❌ File not found."))
            return

        try:
            # Logic branch based on extension
            if path.endswith('.parquet'):
                self._inspect_parquet(path)
            elif path.endswith('.db') or path.endswith('.sqlite'):
                self._inspect_sqlite(path)
            else:
                self.list_tables.addItem(self.tr("⚠️ Unsupported file format."))
                    
        except Exception as e:
            self.list_tables.addItem(f"❌ Error reading file: {e}")

    def _inspect_parquet(self, path):
        """Reads Parquet metadata/columns using Pandas."""
        try:
            # Read just the columns or a small sample to be fast
            df = pd.read_parquet(path)
            
            if 'source_table' in df.columns:
                sources = df['source_table'].unique()
                for s in sources:
                    self.list_tables.addItem(f"📦 [Source] {s}")
                self.list_tables.addItem(f"\nTotal Sources: {len(sources)}")
            else:
                self.list_tables.addItem(f"📄 [Single Table] {os.path.basename(path)}")
                self.list_tables.addItem(f"   Columns: {', '.join(df.columns)}")
                
            self.list_tables.addItem(f"   Total Rows: {len(df)}")
            
        except Exception as e:
            self.list_tables.addItem(f"❌ Failed to inspect Parquet: {e}")

    def _inspect_sqlite(self, path):
        """Legacy SQLite inspection."""
        try:
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view');")
            tables = [r[0] for r in cursor.fetchall() if not r[0].startswith('sqlite_')]
            conn.close()
            
            if not tables:
                self.list_tables.addItem(self.tr("⚠️ No tables found! Invalid DB."))
            else:
                for t in tables:
                    self.list_tables.addItem(f"🗃️ [Table] {t}")
        except Exception as e:
            raise e

class ProcessingPage(QWizardPage):
    def __init__(self, wizard):
        super().__init__(wizard)
        self.setTitle(self.tr("Processing Import"))
        self.setSubTitle(self.tr("Transforming raw data into Base Database (Parquet)..."))
        
        layout = QVBoxLayout(self)
        
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet(ThemeManager.get_style("console_log"))
        layout.addWidget(self.log_box)
        
        # Worker
        self.importer = DatabaseImporter()
        self.thread = QThread()
        self.importer.moveToThread(self.thread)
        
        # Wiring
        self.importer.log_message.connect(self._log)
        self.importer.progress_update.connect(self.progress.setValue)
        self.importer.import_finished.connect(self._on_finished)
        
        # Signal to start
        self.thread.started.connect(self._start_task)

    def initializePage(self):
        # Disable Back button to prevent state corruption during process
        self.wizard().button(QWizard.WizardButton.BackButton).setEnabled(False)
        self.thread.start()

    def _start_task(self):
        path = self.wizard().source_path
        freq = self.wizard().field("target_freq")
        # Trigger ETL
        self.importer.execute_import(path, float(freq))

    @pyqtSlot(str)
    def _log(self, msg):
        self.log_box.append(msg)

    @pyqtSlot(bool, str)
    def _on_finished(self, success, msg):
        self.thread.quit()
        self.thread.wait()
        
        if success:
            self._log(f"\n✅ {msg}")
            self.wizard().button(QWizard.WizardButton.FinishButton).setEnabled(True)
            self.completeChanged.emit() # Notify wizard we are done
        else:
            self._log(f"\n❌ {msg}")
            QMessageBox.critical(self, self.tr("Import Failed"), msg)

    def isComplete(self):
        return not self.thread.isRunning()