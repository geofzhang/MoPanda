import sys
import lasio
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QGridLayout, QVBoxLayout,
                             QPushButton, QFileDialog, QWidget, QInputDialog, QDialog, QFormLayout, QLineEdit,
                             QDialogButtonBox, QGroupBox, QMessageBox)
from PyQt6.QtGui import QShortcut, QKeySequence, QFont
import numpy as np


class LASApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.df = None
        self.las = None
        self.initUI()

    def initUI(self):
        gridLayout = QGridLayout()

        frameFont = QFont("Tilt Neon")

        # File frame with Load and Save LAS buttons
        fileFrame = QGroupBox("File", self)
        fileFrame.setFont(frameFont)
        fileGridLayout = QGridLayout()

        self.loadButton = QPushButton('Load LAS', self)
        self.loadButton.clicked.connect(self.loadLAS)
        fileGridLayout.addWidget(self.loadButton, 0, 0)

        self.saveButton = QPushButton('Save LAS', self)
        self.saveButton.clicked.connect(self.saveLAS)
        fileGridLayout.addWidget(self.saveButton, 0, 1)

        self.exportExcelButton = QPushButton('Export to Excel', self)
        self.exportExcelButton.clicked.connect(self.exportToExcel)
        fileGridLayout.addWidget(self.exportExcelButton, 0, 2)  # Adjust the row and column indices as needed

        self.importExcelButton = QPushButton('Import from Excel', self)
        self.importExcelButton.clicked.connect(self.importFromExcel)
        fileGridLayout.addWidget(self.importExcelButton, 0, 3)

        fileFrame.setLayout(fileGridLayout)
        gridLayout.addWidget(fileFrame, 0, 0, 1, 4)  # Span 2 columns

        # Edit frame with Add and Delete Column buttons
        editFrame = QGroupBox("Edit", self)
        editFrame.setFont(frameFont)
        editGridLayout = QGridLayout()

        self.addColumnButton = QPushButton('Add Column', self)
        self.addColumnButton.clicked.connect(self.addColumn)
        editGridLayout.addWidget(self.addColumnButton, 0, 0)

        self.delColumnButton = QPushButton('Delete Column', self)
        self.delColumnButton.clicked.connect(self.delColumn)
        editGridLayout.addWidget(self.delColumnButton, 0, 1)

        self.addRowButton = QPushButton('Add Row', self)
        self.addRowButton.clicked.connect(self.addRow)
        editGridLayout.addWidget(self.addRowButton, 0, 2)

        self.delRowButton = QPushButton('Delete Row', self)
        self.delRowButton.clicked.connect(self.delRow)
        editGridLayout.addWidget(self.delRowButton, 0, 3)

        editFrame.setLayout(editGridLayout)
        gridLayout.addWidget(editFrame, 1, 0, 1, 4)  # Span 2 columns

        # Table widget
        self.table = QTableWidget(self)
        self.table.setSelectionMode(QTableWidget.SelectionMode.MultiSelection)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectItems)
        gridLayout.addWidget(self.table, 2, 0, 1, 10)  # Spanning 5 columns

        # Add a shortcut for pasting
        self.shortcutPaste = QShortcut(QKeySequence.StandardKey.Paste, self)
        self.shortcutPaste.activated.connect(self.pasteFromClipboard)

        centralWidget = QWidget()
        centralWidget.setLayout(gridLayout)
        self.setCentralWidget(centralWidget)

        self.setWindowTitle('LAS Editor')
        self.showMaximized()

    def loadLAS(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open LAS File", "", "LAS Files (*.las);;All Files (*)")

        if filePath:
            self.las_filename = ".".join(filePath.split("/")[-1].split(".")[:-1])
            las = lasio.read(filePath)
            df = las.df().reset_index()  # Reset index to make depth a regular column
            self.depth_column_name = df.columns[0]
            self.table.setRowCount(df.shape[0])
            self.table.setColumnCount(df.shape[1])

            for i, (index, row) in enumerate(df.iterrows()):
                for j, value in enumerate(row):
                    self.table.setItem(i, j, QTableWidgetItem(str(value)))

            self.table.setHorizontalHeaderLabels(df.columns.tolist())
            self.las = las
            self.df = df

    def saveLAS(self):
        if hasattr(self, 'las'):
            data_dict = {}
            columns = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
            for j, column in enumerate(columns):
                col_data = []
                for i in range(self.table.rowCount()):
                    value = self.table.item(i, j)
                    if value:  # Check if the cell has an item
                        col_data.append(float(value.text()))
                    else:
                        col_data.append(np.nan)
                data_dict[column] = col_data

            depth_data = data_dict.pop(self.depth_column_name)
            df = pd.DataFrame(data_dict, index=depth_data)
            df.index.name = self.depth_column_name

            self.las.set_data(df)
            filePath, _ = QFileDialog.getSaveFileName(self, "Save LAS File", "", "LAS Files (*.las);;All Files (*)")

            if filePath:
                self.las.write(filePath, version=2)
                QMessageBox.information(self, 'Success', 'LAS file has been saved successfully!')

    def exportToExcel(self):
        if hasattr(self, 'las_filename'):
            default_filename = self.las_filename + ".xlsx"
        else:
            default_filename = ""

        filePath, _ = QFileDialog.getSaveFileName(self, "Export to Excel", default_filename,
                                                  "Excel Files (*.xlsx);;All Files (*)")

        if filePath:
            self.df.to_excel(filePath, index=False, engine='openpyxl')

    def importFromExcel(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Import from Excel", "", "Excel Files (*.xlsx);;All Files (*)")
        if filePath:
            try:
                excel_file_name = ".".join(filePath.split("/")[-1].split(".")[:-1])  # Extract file name without extension

                if not hasattr(self, 'las_filename') or excel_file_name != self.las_filename:
                    choice = QMessageBox.warning(self, "File Mismatch",
                                                 "The Excel file doesn't match the loaded LAS file. Please load a "
                                                 "matching LAS file.",
                                                 QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

                    if choice == QMessageBox.StandardButton.Ok:
                        self.loadLAS()
                        return

                # If the names match or after the correct LAS is loaded, read the Excel into the DataFrame
                df = pd.read_excel(filePath, engine='openpyxl')

                # Assuming depth is the first column
                self.depth_column_name = df.columns[0]
                df.set_index(self.depth_column_name, inplace=True)

                # Update the LAS object's data
                self.las.set_data(df)
                self.df = df.reset_index()  # Reflect the changes in the GUI

                # Update the table
                self.table.setRowCount(self.df.shape[0])
                self.table.setColumnCount(self.df.shape[1])
                for i, (index, row) in enumerate(self.df.iterrows()):
                    for j, value in enumerate(row):
                        self.table.setItem(i, j, QTableWidgetItem(str(value)))
                self.table.setHorizontalHeaderLabels(self.df.columns.tolist())

            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while importing Excel file: {str(e)}")

    def addColumn(self):
        # Create a dialog for the input fields
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Column")

        formLayout = QFormLayout()
        mnemonicEdit = QLineEdit()
        descriptionEdit = QLineEdit()
        unitEdit = QLineEdit()

        formLayout.addRow("Mnemonic:", mnemonicEdit)
        formLayout.addRow("Description:", descriptionEdit)
        formLayout.addRow("Unit:", unitEdit)

        buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttonBox.accepted.connect(dialog.accept)
        buttonBox.rejected.connect(dialog.reject)

        formLayout.addRow(buttonBox)
        dialog.setLayout(formLayout)

        # If OK is clicked, retrieve the values
        if dialog.exec() == QDialog.DialogCode.Accepted:
            colMnemonic = mnemonicEdit.text()
            colDescription = descriptionEdit.text()
            colUnit = unitEdit.text()

            if not colMnemonic:
                return

            new_col = [np.nan] * self.table.rowCount()  # Default values set to NaN

            # Append the new curve to the LAS object
            self.las.append_curve(colMnemonic, new_col, unit=colUnit, descr=colDescription)
            print(self.las.curves)
            colIndex = self.table.columnCount()
            self.table.insertColumn(colIndex)  # Add to table widget
            self.table.setHorizontalHeaderItem(colIndex, QTableWidgetItem(colMnemonic))

            for i, value in enumerate(new_col):
                self.table.setItem(i, colIndex, QTableWidgetItem(str(value)))

    def delColumn(self):
        selectedColumns = list(set([index.column() for index in self.table.selectedIndexes()]))

        # Looping in reverse order to avoid messing up column indexes after each deletion
        for colIndex in sorted(selectedColumns, reverse=True):
            if colIndex >= 0:
                colName = self.table.horizontalHeaderItem(colIndex).text()
                self.las.delete_curve(mnemonic=colName)  # Remove from LAS object
                self.table.removeColumn(colIndex)  # Remove from table widget

    def addRow(self):
        selectedRows = list(set([index.row() for index in self.table.selectedIndexes()]))

        # If no row is selected, append at the end
        if not selectedRows:
            rowIndex = self.table.rowCount()
        else:
            rowIndex = max(selectedRows) + 1  # Add after the last selected row

        self.table.insertRow(rowIndex)

        for colIndex in range(self.table.columnCount()):
            self.table.setItem(rowIndex, colIndex, QTableWidgetItem(""))

    def delRow(self):
        selectedRows = list(set([index.row() for index in self.table.selectedIndexes()]))
        # Looping in reverse order to avoid messing up row indexes after each deletion
        for rowIndex in sorted(selectedRows, reverse=True):
            self.table.removeRow(rowIndex)

    def pasteFromClipboard(self):
        clipboard = QApplication.clipboard()
        data = clipboard.text().strip()  # Remove trailing whitespace and newlines

        # Find the starting cell
        startRow = self.table.currentRow()
        startCol = self.table.currentColumn()

        # If no cell is selected, default to (0, 0)
        if startRow == -1 or startCol == -1:
            startRow, startCol = 0, 0

        rows = data.split('\n')
        for i, row in enumerate(rows):
            cells = row.split('\t')
            for j, cell in enumerate(cells):
                # Calculate the actual row and column index
                rowIndex = startRow + i
                colIndex = startCol + j

                # Expand table if necessary
                if rowIndex >= self.table.rowCount():
                    self.table.insertRow(self.table.rowCount())
                if colIndex >= self.table.columnCount():
                    self.table.insertColumn(self.table.columnCount())

                self.table.setItem(rowIndex, colIndex, QTableWidgetItem(cell))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LASApp()
    sys.exit(app.exec())
