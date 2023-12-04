import sys
import os
import lasio
import json
import csv
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QGridLayout, QVBoxLayout,
                             QHBoxLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QColorDialog, QGraphicsPathItem,
                             QPushButton, QFileDialog, QWidget, QInputDialog, QDialog, QFormLayout, QLineEdit,
                             QDialogButtonBox, QGroupBox, QMessageBox, QTabWidget, QLabel, QSplitter, QComboBox,
                             QSlider)
from PyQt6.QtGui import QShortcut, QKeySequence, QColor, QFont, QPixmap, QImage, QPainter, QPainterPath
import numpy as np
import matplotlib.pyplot as plt
import math
from pyqtgraph import BusyCursor
import matplotlib.cm
from datetime import datetime

## Switch to using white background and black foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class ColorPaletteDialog(QDialog):
    def __init__(self, parent=None):
        """
        Initialize the color palette dialog.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super(ColorPaletteDialog, self).__init__(parent)

        self.selectedColor = None

        layout = QGridLayout(self)

        colors = ['0C5DA5', '00B945', 'FF9500', 'FF2C00', '845B97', '474747', '9e9e9e']  # Standard Color Cycle

        for i, color in enumerate(colors):
            btn = QPushButton(self)
            btn.setStyleSheet(f"background-color: {color};")
            btn.clicked.connect(lambda col=color: self.colorSelected(col))
            layout.addWidget(btn, i // 3, i % 3)

    def colorSelected(self, color):
        self.selectedColor = color
        self.accept()


class LASApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.depthTextItems = []
        self.depth_curve_name = "DEPT"
        self.df = None
        self.las = None
        self.track_columns = {
            "CALIPER": ["CAL_N"],
            "DEPTH": ["DEPT"],
            "RESISTIVITY": ["RESSHAL_N", "RESDEEP_N"],
            "GAMMA RAY": ["SGR_N", "CGR_N"],
            "DENSITY": ["RHOB_N"],
            "PE": ["PE_N"],
            "POROSITY": ["PHIE", "DPHI_N", "NPHI_N"],
            "SONIC": ["DTC_N", "DTS_N"],
            "PERMEABILITY": ["K_SDR", "K_TIM"],
            "SALINITY": ["SALINITY", "RWA_N"],
            "FACIES": ["FACIES_KMEANS"],
            "OTHERS": [],

        }
        self.valid_color_selection = ["color", "left_cut_off_color", "right_cut_off_color"]
        self.pen_styles = {
            'Solid': Qt.PenStyle.SolidLine,
            'Dash': Qt.PenStyle.DashLine,
            'Dot': Qt.PenStyle.DotLine,
            'Dash-Dot': Qt.PenStyle.DashDotLine,
            'Dash-Dot-Dot': Qt.PenStyle.DashDotDotLine
        }

        self.palettes = ["copper", "viridis", "plasma", "cividis", "inferno", "magma", "cubehelix", "Paired", "tab20"]
        self.all_parameters = {}
        self.track_parameters = {}
        self.curve_parameters = {}
        self.track_plot_items = {}
        self.trackCount = 0
        self.total_TrackWidth = 0

        # initialize the main GUI
        self.initUI()

    def initUI(self):
        """
        The initUI function is responsible for setting up the main window of the application.
        It creates a QGridLayout, which is then populated with widgets and layouts. The layout
        is then set as the central widget of the main window.

        :param self: Refer to the object itself
        :return: The mainlayout
        :doc-author: Trelent
        """
        mainLayout = QGridLayout()

        # Create a Splitter for vertical division
        verticalSplitter = QSplitter(Qt.Orientation.Vertical)

        # Create a Splitter for horizontal division
        horizontalSplitter = QSplitter(Qt.Orientation.Horizontal)

        # Create a main GraphicsLayout
        self.mainGraphicsLayout = pg.GraphicsLayout()

        # Add the mainGraphicsLayout to plotWidget
        self.plotWidget = pg.GraphicsView()  # Use GraphicsView to show the mainGraphicsLayout
        self.plotWidget.setCentralItem(self.mainGraphicsLayout)
        self.mainGraphicsLayout.layout.setSpacing(0)
        horizontalSplitter.addWidget(self.plotWidget)

        self.currentFileNameTextBox = QLineEdit("No LAS file loaded", self)
        self.currentFileNameTextBox.setReadOnly(True)
        self.currentFileNameTextBox.setStyleSheet("border: 2px solid black; font-weight: bold;")
        mainLayout.addWidget(self.currentFileNameTextBox)

        # Tabs setup
        tabs = QTabWidget(self)

        # File Tab
        fileTab = QWidget(self)
        fileGridLayout = QGridLayout(fileTab)

        self.load_las_Button = QPushButton('Load LAS', self)
        self.load_las_Button.clicked.connect(self.loadLAS)
        fileGridLayout.addWidget(self.load_las_Button, 0, 0)

        self.save_las_Button = QPushButton('Save LAS', self)
        self.save_las_Button.clicked.connect(self.saveLAS)
        fileGridLayout.addWidget(self.save_las_Button, 0, 1)

        self.load_xml_button = QPushButton('Load XML Template', self)
        self.load_xml_button.clicked.connect(self.load_xml)
        fileGridLayout.addWidget(self.load_xml_button, 1, 0)

        self.exportExcelButton = QPushButton('Export to Excel', self)
        self.exportExcelButton.clicked.connect(self.exportToExcel)
        fileGridLayout.addWidget(self.exportExcelButton, 2, 0)

        self.importExcelButton = QPushButton('Import from Excel', self)
        self.importExcelButton.clicked.connect(self.importFromExcel)
        fileGridLayout.addWidget(self.importExcelButton, 2, 1)

        tabs.addTab(fileTab, "File")

        # Edit Table Tab
        editTab = QWidget(self)
        editGridLayout = QGridLayout(editTab)

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

        tabs.addTab(editTab, "Edit")

        # Display Tab
        displayTab = QWidget(self)
        displayLayout = QVBoxLayout(displayTab)

        # GroupBox for Load Tracks
        loadTracksGroupBox = QGroupBox("Load Tracks", self)
        loadTracksLayout = QVBoxLayout(loadTracksGroupBox)

        # Create a new horizontal layout for the ComboBox and the buttons
        trackControlsLayout = QHBoxLayout()

        # Dropdown for adding tracks
        self.trackDropdown = QComboBox(self)
        tracks = list(self.track_columns.keys())
        self.trackDropdown.addItems(tracks)
        trackControlsLayout.addWidget(self.trackDropdown)

        # Buttons for adding and removing tracks
        self.addTrackButton = QPushButton("Add Track", self)
        self.addTrackButton.clicked.connect(lambda checked=False: self.addTrack())
        trackControlsLayout.addWidget(self.addTrackButton)

        self.removeTrackButton = QPushButton("Remove Track", self)
        self.removeTrackButton.clicked.connect(self.removeTrack)
        trackControlsLayout.addWidget(self.removeTrackButton)

        self.removeAllTracksButton = QPushButton('Remove All Tracks', self)
        self.removeAllTracksButton.clicked.connect(self.removeAllTracks)
        trackControlsLayout.addWidget(self.removeAllTracksButton)

        # Add trackControlsLayout to the main loadTracksLayout
        loadTracksLayout.addLayout(trackControlsLayout)

        # Create and configure the QLineEdit and QPushButton for saving/loading templates
        self.currentTemplateTextbox = QLineEdit(self)
        self.currentTemplateTextbox.setReadOnly(True)
        self.currentTemplateTextbox.setPlaceholderText("Currently loaded template...")

        self.saveTemplateButton = QPushButton("Save Template", self)
        self.saveTemplateButton.clicked.connect(self.onSaveButtonClicked)

        self.loadTemplateButton = QPushButton("Load Template", self)
        self.loadTemplateButton.clicked.connect(self.onLoadButtonClicked)

        templateLayout = QHBoxLayout()
        templateLayout.addWidget(self.currentTemplateTextbox)
        templateLayout.addWidget(self.saveTemplateButton)
        templateLayout.addWidget(self.loadTemplateButton)

        loadTracksLayout.addLayout(templateLayout)

        # Add the Load Tracks group box to the Display layout
        displayLayout.addWidget(loadTracksGroupBox)

        # GroupBox for Format Tracks
        self.formatTracksGroupBox = QGroupBox("Format Tracks", self)
        self.formatTracksLayout = QVBoxLayout(self.formatTracksGroupBox)

        # Tab widget for individual tracks inside Format Tracks
        self.tracksTabWidget = QTabWidget(self)
        self.formatTracksLayout.addWidget(self.tracksTabWidget)

        # Add the Format Tracks group box to the Display layout
        displayLayout.addWidget(self.formatTracksGroupBox)

        # Add the Display Tab to the main layout
        tabs.addTab(displayTab, "Display")

        # Tops Tab
        topsTab = QWidget(self)
        topsLayout = QVBoxLayout(topsTab)

        # First Row of Buttons: Load and Save Tops
        topsFirstRowLayout = QHBoxLayout()
        loadTopsButton = QPushButton('Load Tops', self)
        loadTopsButton.clicked.connect(self.loadTops)  # Placeholder for the loadTops method
        saveTopsButton = QPushButton('Save Tops', self)
        saveTopsButton.clicked.connect(self.saveTops)  # Placeholder for the saveTops method
        topsFirstRowLayout.addWidget(loadTopsButton)
        topsFirstRowLayout.addWidget(saveTopsButton)
        topsLayout.addLayout(topsFirstRowLayout)

        # Second Row of Buttons: Add and Delete Top
        topsSecondRowLayout = QHBoxLayout()
        addTopButton = QPushButton('Add Top', self)
        addTopButton.clicked.connect(self.addTop)  # Placeholder for the addTop method
        deleteTopButton = QPushButton('Delete Top', self)
        deleteTopButton.clicked.connect(self.deleteTop)  # Placeholder for the deleteTop method
        topsSecondRowLayout.addWidget(addTopButton)
        topsSecondRowLayout.addWidget(deleteTopButton)
        topsLayout.addLayout(topsSecondRowLayout)

        # Third Row of Buttons: Add and Delete Top from plot
        topsThirdRowLayout = QHBoxLayout()
        drawTopsButton = QPushButton('Draw Tops', self)
        drawTopsButton.clicked.connect(self.initialDrawTops)  # Placeholder for the addTop method
        removeTopsButton = QPushButton('Remove Tops from Display', self)
        removeTopsButton.clicked.connect(self.removeTops)  # Placeholder for the deleteTop method
        topsThirdRowLayout.addWidget(drawTopsButton)
        topsThirdRowLayout.addWidget(removeTopsButton)
        topsLayout.addLayout(topsThirdRowLayout)

        # Bottom Section with Table and Color Buttons
        topsTableLayout = QVBoxLayout()
        self.topsTable = QTableWidget(self)
        topsTableLayout.addWidget(self.topsTable)
        topsLayout.addLayout(topsTableLayout)

        # Add the Tops tab to the main tabs
        tabs.addTab(topsTab, "Tops")

        # Adding tabs and plot to the horizontal splitter
        horizontalSplitter.addWidget(self.currentFileNameTextBox)
        horizontalSplitter.addWidget(tabs)
        horizontalSplitter.addWidget(self.plotWidget)

        # Table widget
        self.table = QTableWidget(self)
        self.table.setSelectionMode(QTableWidget.SelectionMode.MultiSelection)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectItems)

        # Add a shortcut for pasting
        self.shortcutPaste = QShortcut(QKeySequence.StandardKey.Paste, self)
        self.shortcutPaste.activated.connect(self.pasteFromClipboard)

        # Adding horizontal splitter and table to the vertical splitter
        verticalSplitter.addWidget(horizontalSplitter)
        verticalSplitter.addWidget(self.table)

        mainLayout.addWidget(verticalSplitter)

        centralWidget = QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

        self.setWindowTitle('LAS Editor')
        self.showMaximized()

    def addTrack(self, params=None):
        """
        Adds a new track to the tracks tab widget.

        If the selected track name from the dropdown is "OTHERS", prompts the user to enter a custom track name.
        Validates the custom track name to ensure it is not empty, not already used, and strips any whitespace.
        If the track name is valid and new, it creates a new tab with various controls for track manipulation and log loading.

        Parameters:
        - params (dict, optional): A dictionary of parameters to be used for creating track parameters. Defaults to None.

        Returns:
        None
        """
        trackName = self.trackDropdown.currentText()
        if trackName == "OTHERS":
            # Prompt the user to enter a custom track name
            text, ok = QInputDialog.getText(self, 'Input Track Name', 'Enter the custom track name:')
            if ok and text:
                trackName = text.strip()  # Remove any leading/trailing whitespace
                if trackName and trackName not in self.track_columns:
                    # Update track_columns and dropdown
                    self.track_columns[trackName] = []  # Initialize with an empty list or whatever is appropriate
                    self.trackDropdown.addItem(trackName)
                else:
                    QMessageBox.warning(self, "Warning", "Track name already exists or is invalid.")
                    return
            else:
                # User canceled or did not enter a name, do not proceed
                return

        if not self.isTrackAlreadyAdded(trackName):
            trackTab = QWidget()
            tabLayout = QVBoxLayout(trackTab)  # This is the main layout of the tab

            # Create the track controls (Move forward, Move backward, Remove Track)
            self.createTrackControls(tabLayout)

            # Create the Track Parameters GroupBox
            paramsGroupBox = self.createTrackParameters(trackName, params)
            tabLayout.addWidget(paramsGroupBox)

            # Create a horizontal layout to hold both buttons
            buttonsLayout = QHBoxLayout()

            # Create Auto Load and Manually Add Logs Buttons and add them to the horizontal layout
            autoLoadLogsButton = QPushButton("Auto Load Logs")
            autoLoadLogsButton.clicked.connect(self.autoLoadLogs)
            buttonsLayout.addWidget(autoLoadLogsButton)

            manuallyAddLogsButton = QPushButton("Manually Add Logs")
            manuallyAddLogsButton.clicked.connect(self.manuallyAddLogs)
            buttonsLayout.addWidget(manuallyAddLogsButton)

            # Add the horizontal layout containing buttons to the main layout
            tabLayout.addLayout(buttonsLayout)

            tabLayout.addStretch(1)  # Ensure this is added to the main layout of the tab

            trackIndex = self.tracksTabWidget.addTab(trackTab, trackName)
            self.trackCount += 1
            self.plot_track(trackName, trackIndex)
            self.updateTrackWidths()

    def removeTrack(self):
        """
        The removeTrack function removes a track from the GUI.
        It does this by removing the tab for that track, and then removing all of its parameters from self.track_parameters and self.all_parameters.

        :param self: Refer to the object that is calling the function
        :return: The index of the track that was removed
        """
        trackName = self.trackDropdown.currentText()
        index = -1
        for i in range(self.tracksTabWidget.count()):
            if self.tracksTabWidget.tabText(i) == trackName:
                index = i
                break
        if index != -1:
            del self.track_parameters[trackName]
            del self.all_parameters[trackName]
            self.tracksTabWidget.removeTab(index)
            # Remove the plot for this track
            self.removePlotForTrack(trackName, index)
            self.trackCount -= 1
            self.updateTrackWidths()

    def removeAllTracks(self):
        # Clear all track parameters
        self.track_parameters.clear()
        self.all_parameters.clear()

        # Loop through each track and apply necessary cleanup
        for i in range(self.tracksTabWidget.count()):
            trackName = self.tracksTabWidget.tabText(i)
            self.removePlotForTrack(trackName, i)

        # Remove all tabs from tracksTabWidget
        while self.tracksTabWidget.count() > 0:
            self.tracksTabWidget.removeTab(0)

        # Clear the corresponding plots
        self.mainGraphicsLayout.clear()

        # Reset track count and update widths
        self.trackCount = 0
        self.updateTrackWidths()

    def moveForward(self):
        """
        The moveForward function moves the current track one position forward in the tracksTabWidget.

        :param self: Refer to the object itself
        :return: The following:
        """
        currentIndex = self.tracksTabWidget.currentIndex()
        if currentIndex > 0:  # Can move forward
            # Get track names
            currentTrackName = self.tracksTabWidget.tabText(currentIndex)
            previousTrackName = self.tracksTabWidget.tabText(currentIndex - 1)

            # Swap the contents of the dictionaries
            self.track_parameters[currentTrackName], self.track_parameters[previousTrackName] = \
                self.track_parameters[previousTrackName], self.track_parameters[currentTrackName]
            self.all_parameters[currentTrackName], self.all_parameters[previousTrackName] = \
                self.all_parameters[previousTrackName], self.all_parameters[currentTrackName]

            # Rename the keys
            self.track_parameters = {
                previousTrackName if k == currentTrackName else currentTrackName if k == previousTrackName else k: v for
                k, v in self.track_parameters.items()}
            self.all_parameters = {
                previousTrackName if k == currentTrackName else currentTrackName if k == previousTrackName else k: v for
                k, v in self.all_parameters.items()}

            # Swap plots
            self.swapTracks(currentIndex, currentIndex - 1)

            # Swap tabs
            currentTab = self.tracksTabWidget.widget(currentIndex)
            currentTabText = self.tracksTabWidget.tabText(currentIndex)
            self.tracksTabWidget.removeTab(currentIndex)
            self.tracksTabWidget.insertTab(currentIndex - 1, currentTab, currentTabText)
            self.tracksTabWidget.setCurrentIndex(currentIndex - 1)

    def moveBackward(self):
        currentIndex = self.tracksTabWidget.currentIndex()
        if currentIndex < self.tracksTabWidget.count() - 1:  # Can move backward
            # Get track names
            currentTrackName = self.tracksTabWidget.tabText(currentIndex)
            nextTrackName = self.tracksTabWidget.tabText(currentIndex + 1)

            # Swap in self.track_parameters and self.all_parameters
            self.track_parameters[currentTrackName], self.track_parameters[nextTrackName] = \
                self.track_parameters[nextTrackName], self.track_parameters[currentTrackName]
            self.all_parameters[currentTrackName], self.all_parameters[nextTrackName] = \
                self.all_parameters[nextTrackName], self.all_parameters[currentTrackName]

            # Rename the keys
            self.track_parameters = {
                nextTrackName if k == currentTrackName else currentTrackName if k == nextTrackName else k: v for k, v in
                self.track_parameters.items()}
            self.all_parameters = {
                nextTrackName if k == currentTrackName else currentTrackName if k == nextTrackName else k: v for k, v in
                self.all_parameters.items()}

            # Swap plots
            self.swapTracks(currentIndex, currentIndex + 1)

            # Swap tabs
            currentTab = self.tracksTabWidget.widget(currentIndex)
            currentTabText = self.tracksTabWidget.tabText(currentIndex)
            self.tracksTabWidget.removeTab(currentIndex)
            self.tracksTabWidget.insertTab(currentIndex + 1, currentTab, currentTabText)
            self.tracksTabWidget.setCurrentIndex(currentIndex + 1)

    def removeTrackintab(self):
        currentIndex = self.tracksTabWidget.currentIndex()
        trackName = self.tracksTabWidget.tabText(currentIndex)
        del self.track_parameters[trackName]
        del self.all_parameters[trackName]
        self.tracksTabWidget.removeTab(currentIndex)
        self.removePlotForTrack(trackName, currentIndex)
        self.trackCount -= 1
        self.updateTrackWidths()

    def isTrackAlreadyAdded(self, trackName):
        for index in range(self.tracksTabWidget.count()):
            if self.tracksTabWidget.tabText(index) == trackName and trackName != "OTHERS":
                return True
        return False

    def createTrackControls(self, tab):
        layout = QHBoxLayout()  # Adjust layout type as needed
        moveForwardBtn = QPushButton("Move Forward")
        moveBackwardBtn = QPushButton("Move Backward")
        removeTrackBtn = QPushButton("Remove Track")

        moveForwardBtn.clicked.connect(self.moveForward)
        moveBackwardBtn.clicked.connect(self.moveBackward)
        removeTrackBtn.clicked.connect(self.removeTrackintab)

        layout.addWidget(moveForwardBtn)
        layout.addWidget(moveBackwardBtn)
        layout.addWidget(removeTrackBtn)

        # Create the main vertical layout and add the button layout
        mainLayout = QVBoxLayout()
        mainLayout.addLayout(layout)
        mainLayout.addStretch()  # This will push the button layout to the top

        tab.addLayout(mainLayout)

    def createTrackParameters(self, trackName, params=None):
        if params is None:
            params = {
                "track_display_name": trackName,
                "scale": "normal",
                "track_width": 1,
                "major_line_width": 3,
                "number_spacing": 50,
                "tick_spacing": 10,
                "line_spacing": 50
            }

        paramsGroupBox = QGroupBox("Track Parameters")
        paramsLayout = QGridLayout(paramsGroupBox)

        # Track Display Name and Scale
        track_display_name = QLineEdit(params["track_display_name"])
        track_display_name.textChanged.connect(lambda value: self.updateTrackDisplayName(trackName, value))
        scale = QComboBox()
        scale.addItems(["normal", "log scale"])
        if trackName in ["RESISTIVITY", "PERMEABILITY"]:
            scale.setCurrentIndex(1)
        scale.currentTextChanged.connect(lambda value: self.updateScale(trackName, value))

        paramsLayout.addWidget(QLabel("Track Display Name:"), 0, 0)
        paramsLayout.addWidget(track_display_name, 0, 1, 1, 5)
        paramsLayout.addWidget(QLabel("Scale:"), 1, 0)
        paramsLayout.addWidget(scale, 1, 1)

        # Track Width and Major Line Width
        track_width = QDoubleSpinBox()
        track_width.setMaximum(5.00)
        track_width.setMinimum(0.00)
        track_width.setValue(params["track_width"])
        track_width.valueChanged.connect(lambda value: self.updateTrackWidth(trackName, value))

        major_line_width = QDoubleSpinBox()
        major_line_width.setValue(params["major_line_width"])
        major_line_width.valueChanged.connect(lambda value: self.updateMajorLineWidth(trackName, value))

        paramsLayout.addWidget(QLabel("Track Width:"), 1, 2)
        paramsLayout.addWidget(track_width, 1, 3)
        paramsLayout.addWidget(QLabel("Major Line Width:"), 1, 4)
        paramsLayout.addWidget(major_line_width, 1, 5)

        self.track_parameters[trackName] = {
            "track_display_name": track_display_name.text(),
            "scale": scale.currentText(),
            "track_width": track_width.value(),
            "major_line_width": major_line_width.value()
        }

        if trackName == "DEPTH":
            number_spacing = QDoubleSpinBox()
            number_spacing.setMaximum(1e6)
            number_spacing.setValue(params["number_spacing"])
            number_spacing.editingFinished.connect(lambda: self.updateNumberSpacing(trackName, number_spacing.value()))

            tick_spacing = QDoubleSpinBox()
            tick_spacing.setMaximum(1e6)
            tick_spacing.setValue(params["tick_spacing"])
            tick_spacing.editingFinished.connect(lambda: self.updateTickSpacing(trackName, tick_spacing.value()))

            line_spacing = QDoubleSpinBox()
            tick_spacing.setMaximum(1e6)
            line_spacing.setValue(params["line_spacing"])
            line_spacing.editingFinished.connect(lambda: self.updateLineSpacing(trackName, line_spacing.value()))

            paramsLayout.addWidget(QLabel("Number Spacing:"), 2, 0)
            paramsLayout.addWidget(number_spacing, 2, 1)
            paramsLayout.addWidget(QLabel("Tick Spacing:"), 2, 2)
            paramsLayout.addWidget(tick_spacing, 2, 3)
            paramsLayout.addWidget(QLabel("Line Spacing:"), 2, 4)
            paramsLayout.addWidget(line_spacing, 2, 5)

            self.track_parameters[trackName].update({
                "number_spacing": number_spacing.value(),
                "tick_spacing": tick_spacing.value(),
                "line_spacing": line_spacing.value()
            })

        self.all_parameters[trackName] = self.track_parameters[trackName]

        return paramsGroupBox

    def onSaveButtonClicked(self):
        # Prompt the user for a save file location
        """
        The onSaveButtonClicked function is called when the user clicks on the &quot;Save&quot; button.
        It prompts the user for a save file location, and then saves all of the parameters in
        the GUI to that file.

        :param self: Represent the instance of the class
        :return: The filename
        :doc-author: Trelent
        """
        filename, _ = QFileDialog.getSaveFileName(self, "Save Parameters", "", "JSON Files (*.json);;All Files (*)")
        if filename:  # If user didn't cancel
            if not filename.endswith('.json'):
                filename += '.json'  # Ensure it has the right extension
            self.saveParametersToFile(filename)
            template_name = os.path.basename(filename)
            self.currentTemplateTextbox.setText(template_name)

    def onLoadButtonClicked(self):
        # Prompt the user for a load file location
        filename, _ = QFileDialog.getOpenFileName(self, "Load Parameters", "", "JSON Files (*.json);;All Files (*)")
        if filename:  # If user didn't cancel
            self.loadParametersFromFile(filename)
            template_name = os.path.basename(filename)
            self.currentTemplateTextbox.setText(template_name)

    def saveParametersToFile(self, filename):
        try:
            with open(filename, 'w') as file:
                json.dump(self.all_parameters, file, indent=4)  # 4 spaces of indentation
            QMessageBox.information(self, "Success", "Parameters saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while saving parameters: {str(e)}")

    def loadParametersFromFile(self, filename):
        try:
            with BusyCursor():
                with open(filename, 'r') as file:
                    loaded_data = json.load(file)

                # Update self.track_columns and dropdown with tracks from the loaded file
                for trackName in loaded_data.keys():
                    if trackName not in self.track_columns:
                        self.track_columns[trackName] = []  # Add with an empty list or proper defaults
                        self.trackDropdown.addItem(trackName)  # Add to dropdown if not already present

                for trackName, trackData in loaded_data.items():
                    # Check if there is at least one valid curve in the track before adding
                    valid_curves = [col for col in trackData.get("curves", {}) if col in self.curve_exist]

                    if valid_curves:  # Only proceed if there are valid curves
                        # Recreate track if it doesn't exist
                        if not self.isTrackAlreadyAdded(trackName):
                            self.trackDropdown.setCurrentText(trackName)
                            self.addTrack(loaded_data[trackName])

                        # Initialize or retrieve track parameters
                        self.all_parameters[trackName] = self.track_parameters.get(trackName, {})
                        self.all_parameters[trackName]['curves'] = {}

                        for col in valid_curves:
                            curveData = trackData["curves"][col]
                            self.loadLogs(trackName, [col], curveData)
                            self.all_parameters[trackName]["curves"][col] = self.curve_parameters.get(col, {})
                    else:
                        # No valid curves for this track, so we skip adding the track entirely
                        continue

            # Inform the user of success
            QMessageBox.information(self, "Success", "Parameters loaded successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load parameters: {e}")

    def autoLoadLogs(self):
        trackName = self.tracksTabWidget.tabText(self.tracksTabWidget.currentIndex())

        columns_in_table = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
        found_columns = [col for col in self.track_columns.get(trackName, []) if col in columns_in_table]

        # Load the found logs
        self.loadLogs(trackName, found_columns)

    def manuallyAddLogs(self):
        # Retrieve all column headers from the table
        columns_in_table = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]

        # Let the user select multiple columns using getItems
        selected_columns, ok = QInputDialog.getItem(
            self,
            "Select Columns",
            "Choose the columns to load:",
            columns_in_table,
            0,  # current index
            False  # no editing
        )

        if ok and selected_columns:
            trackName = self.tracksTabWidget.tabText(self.tracksTabWidget.currentIndex())

            # Load the selected logs
            self.loadLogs(trackName, [selected_columns])

    def loadLogs(self, trackName, columns, params=None):

        # Update the track tab
        self.updateTrackTabWithColumns(trackName, columns, params)
        # Update the main storage dictionary
        if trackName not in self.all_parameters:
            self.all_parameters[trackName] = self.track_parameters.get(trackName, {})

        # Integrate curve parameters into the main storage dictionary
        if 'curves' not in self.all_parameters[trackName]:
            self.all_parameters[trackName]['curves'] = {}
        for col in columns:
            self.all_parameters[trackName]['curves'][col] = self.curve_parameters.get(col, {})

            # add logs to the plot
            self.plot_curve(trackName, col)

    def updateTrackTabWithColumns(self, trackName, columns, params=None):
        # Find the tab corresponding to the trackName
        index = -1
        for i in range(self.tracksTabWidget.count()):
            if self.tracksTabWidget.tabText(i) == trackName:
                index = i
                break

        if index != -1:
            trackTab = self.tracksTabWidget.widget(index)

            # Check if the trackTab already has a layout, if not, create one.
            if not trackTab.layout():
                trackLayout = QVBoxLayout(trackTab)
                trackTab.setLayout(trackLayout)
            else:
                trackLayout = trackTab.layout()

            # Try to find an existing QTabWidget in the trackLayout
            curvesTabWidget = None
            for i in range(trackLayout.count()):
                widget = trackLayout.itemAt(i).widget()
                if isinstance(widget, QTabWidget):
                    curvesTabWidget = widget
                    break

            # If no existing QTabWidget is found, create one and add to layout
            if not curvesTabWidget:
                curvesTabWidget = QTabWidget(trackTab)
                trackLayout.addWidget(curvesTabWidget)

            # For each column in columns, create a new Tab if it doesn't exist
            for col in columns:
                if curvesTabWidget.indexOf(curvesTabWidget.findChild(QWidget, col)) == -1:  # Check if tab exists
                    curveTab = QWidget()
                    curveTab.setObjectName(col)  # Set object name for finding it later
                    curveLayout = QVBoxLayout(curveTab)

                    # Add the widgets for the curve
                    curveGroupBox = self.createCurveTab(col, trackName, params)
                    curveLayout.addWidget(curveGroupBox)
                    curveTab.setLayout(curveLayout)

                    # Add this curveTab to the curvesTabWidget
                    curvesTabWidget.addTab(curveTab, col)

            # Request update
            trackTab.update()

    def createCurveTab(self, col, trackName, params=None):
        if params is None:
            params = {
                "display_name": col,
                "line_color": "#000000",
                "line_alpha": 255,
                "line_style": "Solid",
                "fill_color_palette": "viridis",
                "filling_side": "None",
                "min_val": 0,
                "max_val": 1,
                "reverse": False,
                "left_cut_off": 0,
                "left_cut_off_color": "#cde7f0",
                "left_cut_off_alpha": 0,
                "right_cut_off": 0,
                "right_cut_off_color": "#ff9994",
                "right_cut_off_alpha": 0,
            }
        # New tab for the column
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Curve Display Name
        displayNameLayout = QHBoxLayout()
        displayNameLayout.addWidget(QLabel("Curve Display Name:"))
        displayName = QLineEdit(params["display_name"])
        displayName.setProperty("column_name", col)  # set the column name as a property
        displayName.setProperty("track_name", trackName)
        displayName.textChanged.connect(
            self.onDisplayNameChanged)  # connect signal to slot displayNameLayout.addWidget(displayName)
        displayNameLayout.addWidget(displayName)

        # Add Remove Curve button
        removeCurveBtn = QPushButton("Remove Curve")
        removeCurveBtn.clicked.connect(lambda: self.removeCurveTab(col, trackName))
        displayNameLayout.addWidget(removeCurveBtn)

        layout.addLayout(displayNameLayout)

        # Color
        colorButton = QPushButton()
        argb_color = self.generate_argb(params['line_color'], params['line_alpha'])
        colorButton.setStyleSheet(f"background-color: {argb_color}; color: #FFFFFF;")
        colorButton.setText(params['line_color'])  # Set the button's text to the color code

        colorAlphaSlider = QSlider(Qt.Orientation.Horizontal)
        colorAlphaSlider.setRange(0, 255)
        colorAlphaSlider.setValue(params["line_alpha"])  # Default to fully opaque
        colorAlphaSlider.setFixedWidth(100)
        colorAlphaSlider.valueChanged.connect(lambda: self.updateButtonTransparency(colorButton, colorAlphaSlider))

        colorlineLayout = QHBoxLayout()
        colorlineLayout.addWidget(QLabel("Line Color:"))
        colorlineLayout.addWidget(colorButton)
        colorlineLayout.addWidget(QLabel("Transparency (0-255):"))
        colorlineLayout.addWidget(colorAlphaSlider)

        colorAlphaSpin = QSpinBox()
        colorAlphaSpin.setRange(0, 255)
        colorAlphaSpin.setValue(params["line_alpha"])  # Default to fully opaque
        colorAlphaSlider.valueChanged.connect(colorAlphaSpin.setValue)
        colorAlphaSpin.valueChanged.connect(colorAlphaSlider.setValue)

        colorButton.clicked.connect(
            lambda: self.openColorDialog(displayName.text(), attribute="line", colorButton=colorButton,
                                         alphaSlider=colorAlphaSlider, context="color"))

        colorAlphaSlider.valueChanged.connect(
            lambda: self.openColorDialog(displayName.text(), attribute="line", colorButton=colorButton,
                                         alphaSlider=colorAlphaSlider, context="alpha_slider"))

        colorAlphaSpin.valueChanged.connect(
            lambda: self.openColorDialog(displayName.text(), attribute="line", colorButton=colorButton,
                                         alphaSlider=colorAlphaSlider, context="alpha_spin"))

        colorlineLayout.addWidget(colorAlphaSpin)

        # Line Style
        colorlineLayout.addWidget(QLabel("Line Style:"))
        lineStyle = QComboBox()
        lineStyles = ["Solid", "Dash", "Dot", "Dash-Dot", "Dash-Dot-Dot"]
        lineStyle.addItems(lineStyles)
        lineStyle.setCurrentText(params["line_style"])
        lineStyle.currentTextChanged.connect(lambda text: self.updateLineStyle(col, trackName, text))
        colorlineLayout.addWidget(lineStyle)

        layout.addLayout(colorlineLayout)

        # Combined Fill Color Palette and Filling Side
        fillSettingsLayout = QHBoxLayout()

        # Fill Color Palette
        fillSettingsLayout.addWidget(QLabel("Fill Color Palette:"))
        fillColorPalette = QComboBox()
        fillColorPalette.addItems(self.palettes)
        fillColorPalette.setCurrentText(params["fill_color_palette"])  # Use the updated params dictionary
        fillColorPalette.currentTextChanged.connect(
            lambda palette: self.updateFillColorPalette(col, trackName, palette))

        fillSettingsLayout.addWidget(fillColorPalette)

        palettePreviewLabel = QLabel()
        currentPalettePixmap = self.generate_palette_preview(fillColorPalette.currentText())
        palettePreviewLabel.setPixmap(currentPalettePixmap)
        fillColorPalette.currentIndexChanged.connect(
            lambda: palettePreviewLabel.setPixmap(self.generate_palette_preview(fillColorPalette.currentText())))
        fillSettingsLayout.addWidget(palettePreviewLabel)

        # Filling Side
        fillSettingsLayout.addWidget(QLabel("Filling Side:"))
        fillingSide = QComboBox()
        sides = ["None", "Left", "Right"]
        fillingSide.addItems(sides)
        fillingSide.setCurrentText(params["filling_side"])  # Use the updated params dictionary
        fillingSide.currentTextChanged.connect(
            lambda side: self.updateFillingSide(col, trackName, side))
        fillSettingsLayout.addWidget(fillingSide)

        layout.addLayout(fillSettingsLayout)

        # Min, Max, and Reverse in one line
        minMaxReverseLayout = QHBoxLayout()

        # Min
        minMaxReverseLayout.addWidget(QLabel("Min:"))
        minVal = QDoubleSpinBox()
        minVal.setMaximum(1e6)
        minVal.setMinimum(-1e6)
        minVal.setDecimals(4)  # Set precision to 4 decimal places
        minVal.setValue(params["min_val"])
        minVal.valueChanged.connect(lambda value: self.updateMinValue(col, trackName, value))
        minMaxReverseLayout.addWidget(minVal)

        # Max
        minMaxReverseLayout.addWidget(QLabel("Max:"))
        maxVal = QDoubleSpinBox()
        maxVal.setMaximum(1e6)
        maxVal.setMinimum(-1e6)
        maxVal.setDecimals(4)  # Set precision to 4 decimal places
        maxVal.setValue(params["max_val"])
        maxVal.valueChanged.connect(lambda value: self.updateMaxValue(col, trackName, value))
        minMaxReverseLayout.addWidget(maxVal)

        # Reverse
        reverseCheckbox = QCheckBox("Reverse")
        reverseCheckbox.setChecked(params["reverse"])
        reverseCheckbox.toggled.connect(lambda checked: self.updateReverse(col, trackName, checked))
        minMaxReverseLayout.addWidget(reverseCheckbox)

        layout.addLayout(minMaxReverseLayout)

        # Left Cut-off Value
        leftCutOffLayout = QHBoxLayout()
        leftCutOffLayout.addWidget(QLabel("Left Cut-off Value:"))
        leftCutOff = QDoubleSpinBox()
        leftCutOff.setMaximum(1e6)
        leftCutOff.setMinimum(-1e6)
        leftCutOff.setValue(params["left_cut_off"])
        leftCutOff.valueChanged.connect(lambda value: self.updateLeftCutOff(col, trackName, value))
        leftCutOffLayout.addWidget(leftCutOff)
        layout.addLayout(leftCutOffLayout)

        # Left Cut-off Color and Transparency Slider
        leftCutOffColorLayout = QHBoxLayout()
        leftCutOffColorLayout.addWidget(QLabel("Left Cut-off Color:"))

        leftCutOffColorButton = QPushButton()
        left_argb_color = self.generate_argb(params['left_cut_off_color'], params['left_cut_off_alpha'])
        leftCutOffColorButton.setStyleSheet(f"background-color: {left_argb_color}; color: #FFFFFF;")
        leftCutOffColorButton.setText(params['left_cut_off_color'])
        leftCutOffColorLayout.addWidget(leftCutOffColorButton)

        # Transparency Slider
        leftCutOffColorAlphaSlider = QSlider(Qt.Orientation.Horizontal)
        leftCutOffColorAlphaSlider.setRange(0, 255)  # 0 is fully transparent and 255 is fully opaque
        leftCutOffColorAlphaSlider.setValue(params["left_cut_off_alpha"])
        leftCutOffColorAlphaSlider.setFixedWidth(100)  # Set a fixed width
        leftCutOffColorLayout.addWidget(QLabel("Transparency (0-255):"))
        leftCutOffColorLayout.addWidget(leftCutOffColorAlphaSlider)

        leftCutOffColorAlphaSlider.valueChanged.connect(
            lambda: self.updateButtonTransparency(leftCutOffColorButton, leftCutOffColorAlphaSlider))

        # Transparency SpinBox for Left Cut-off
        leftAlphaSpinBox = QSpinBox()
        leftAlphaSpinBox.setRange(0, 255)
        leftAlphaSpinBox.setValue(params["left_cut_off_alpha"])
        leftCutOffColorLayout.addWidget(leftAlphaSpinBox)

        # Connect slider and spinbox for two-way sync
        leftCutOffColorAlphaSlider.valueChanged.connect(leftAlphaSpinBox.setValue)
        leftAlphaSpinBox.valueChanged.connect(leftCutOffColorAlphaSlider.setValue)

        leftCutOffColorButton.clicked.connect(
            lambda: self.openColorDialog(displayName.text(), attribute="left_cut_off",
                                         colorButton=leftCutOffColorButton,
                                         alphaSlider=leftCutOffColorAlphaSlider, context="color"))

        leftCutOffColorAlphaSlider.valueChanged.connect(
            lambda: self.openColorDialog(displayName.text(), attribute="left_cut_off",
                                         colorButton=leftCutOffColorButton,
                                         alphaSlider=leftCutOffColorAlphaSlider, context="alpha_slider"))

        leftAlphaSpinBox.valueChanged.connect(
            lambda: self.openColorDialog(displayName.text(), attribute="left_cut_off",
                                         colorButton=leftCutOffColorButton,
                                         alphaSlider=leftCutOffColorAlphaSlider, context="alpha_spin"))
        layout.addLayout(leftCutOffColorLayout)

        # Right Cut-off Value
        rightCutOffLayout = QHBoxLayout()
        rightCutOffLayout.addWidget(QLabel("Right Cut-off Value:"))
        rightCutOff = QDoubleSpinBox()
        rightCutOff.setMaximum(1e6)
        rightCutOff.setMinimum(-1e6)
        rightCutOff.setValue(params["right_cut_off"])
        rightCutOffLayout.addWidget(rightCutOff)
        layout.addLayout(rightCutOffLayout)

        # Right Cut-off Color and Transparency Slider
        rightCutOffColorLayout = QHBoxLayout()
        rightCutOffColorLayout.addWidget(QLabel("Right Cut-off Color:"))

        rightCutOffColorButton = QPushButton()
        right_argb_color = self.generate_argb(params['right_cut_off_color'], params['right_cut_off_alpha'])
        rightCutOffColorButton.setStyleSheet(f"background-color: {right_argb_color}; color: #FFFFFF;")
        rightCutOffColorButton.setText(params['right_cut_off_color'])
        rightCutOff.valueChanged.connect(lambda value: self.updateRightCutOff(col, trackName, value))
        rightCutOffColorLayout.addWidget(rightCutOffColorButton)

        # Transparency Slider
        rightCutOffColorAlphaSlider = QSlider(Qt.Orientation.Horizontal)
        rightCutOffColorAlphaSlider.setRange(0, 255)  # 0 is fully transparent and 255 is fully opaque
        rightCutOffColorAlphaSlider.setValue(params["right_cut_off_alpha"])
        rightCutOffColorAlphaSlider.setFixedWidth(100)
        rightCutOffColorLayout.addWidget(QLabel("Transparency (0-255):"))
        rightCutOffColorLayout.addWidget(rightCutOffColorAlphaSlider)

        rightCutOffColorAlphaSlider.valueChanged.connect(
            lambda: self.updateButtonTransparency(rightCutOffColorButton, rightCutOffColorAlphaSlider))

        # Transparency SpinBox for Right Cut-off
        rightAlphaSpinBox = QSpinBox()
        rightAlphaSpinBox.setRange(0, 255)
        rightAlphaSpinBox.setValue(params["right_cut_off_alpha"])
        rightCutOffColorLayout.addWidget(rightAlphaSpinBox)

        # Connect slider and spinbox for two-way sync
        rightCutOffColorAlphaSlider.valueChanged.connect(rightAlphaSpinBox.setValue)
        rightAlphaSpinBox.valueChanged.connect(rightCutOffColorAlphaSlider.setValue)

        rightCutOffColorButton.clicked.connect(
            lambda: self.openColorDialog(displayName.text(), attribute="right_cut_off",
                                         colorButton=rightCutOffColorButton,
                                         alphaSlider=rightCutOffColorAlphaSlider, context="color"))

        rightCutOffColorAlphaSlider.valueChanged.connect(
            lambda: self.openColorDialog(displayName.text(), attribute="right_cut_off",
                                         colorButton=rightCutOffColorButton,
                                         alphaSlider=rightCutOffColorAlphaSlider, context="alpha_slider"))

        rightAlphaSpinBox.valueChanged.connect(
            lambda: self.openColorDialog(displayName.text(), attribute="right_cut_off",
                                         colorButton=rightCutOffColorButton,
                                         alphaSlider=rightCutOffColorAlphaSlider, context="alpha_spin"))
        layout.addLayout(rightCutOffColorLayout)

        self.curve_parameters[col] = {
            "display_name": displayName.text(),
            "line_color": colorButton.text(),
            "line_alpha": colorAlphaSlider.value(),
            "line_style": lineStyle.currentText(),
            "fill_color_palette": fillColorPalette.currentText(),
            "filling_side": fillingSide.currentText(),
            "min_val": minVal.value(),
            "max_val": maxVal.value(),
            "reverse": reverseCheckbox.isChecked(),
            "left_cut_off": leftCutOff.value(),
            "left_cut_off_color": leftCutOffColorButton.text(),
            "left_cut_off_alpha": leftCutOffColorAlphaSlider.value(),
            "right_cut_off": rightCutOff.value(),
            "right_cut_off_color": rightCutOffColorButton.text(),
            "right_cut_off_alpha": rightCutOffColorAlphaSlider.value(),
        }

        return tab

    ### Updating Functions of the Track and Curve parameters ###

    def on_table_item_changed(self, item):
        # Get the column of the changed item
        column = item.column()
        # Get the header of the column, which corresponds to 'col'
        col_header = self.table.horizontalHeaderItem(column).text()

        # Update the specific curve for each track
        for trackName in self.track_plot_items:
            # Check if 'curves' key and specific column exists before updating
            if 'logs' in self.track_plot_items[trackName] and col_header in self.track_plot_items[trackName][
                'logs']:
                new_curve_data = np.array(self.get_column_data(col_header), dtype=float)
                new_depth_data = np.array(self.get_column_data(self.depth_curve_name), dtype=float)
                self.update_curve(trackName, col_header, new_curve_data, new_depth_data)

    def update_curve(self, trackName, col, new_curve_data, new_depth_data):
        logItem = self.track_plot_items[trackName]['logs'][col]
        logItem.setData(new_curve_data, new_depth_data)

        # Refresh the plot
        plotItem = self.track_plot_items[trackName]['plot']
        plotItem.update()

    def updateTrackDisplayName(self, trackName, value):
        self.track_parameters[trackName]["track_display_name"] = value
        self.all_parameters[trackName]["track_display_name"] = value
        self.updateTitle(trackName, value)

    def updateScale(self, trackName, value):
        self.track_parameters[trackName]["scale"] = value
        self.all_parameters[trackName]["scale"] = value
        plotItem = self.track_plot_items[trackName]["plot"]
        if value == "normal":
            plotItem.setLogMode(x=False, y=False)
        else:
            plotItem.setLogMode(x=True, y=False)

    def updateTrackWidth(self, trackName, value):
        self.track_parameters[trackName]["track_width"] = value
        self.all_parameters[trackName]["track_width"] = value
        self.updateTrackWidths()

    def updateMajorLineWidth(self, trackName, value):
        self.track_parameters[trackName]["major_line_width"] = value
        self.all_parameters[trackName]["major_line_width"] = value
        if trackName == 'DEPTH':
            self._display_depth_track(self.depth_curve_name)
        elif trackName in self.all_parameters:
            for column in self.all_parameters[trackName]["curves"]:
                if 'logs' in self.track_plot_items[trackName] and column in self.track_plot_items[trackName]['logs']:
                    logItem = self.track_plot_items[trackName]['logs'][column]
                    pen = logItem.opts['pen']
                    pen.setWidthF(value)
                    logItem.setPen(pen)

    def updateNumberSpacing(self, trackName, value):
        self.track_parameters[trackName]["number_spacing"] = value
        self.all_parameters[trackName]["number_spacing"] = value
        self._display_depth_track(self.depth_curve_name)

    def updateTickSpacing(self, trackName, value):
        self.track_parameters[trackName]["tick_spacing"] = value
        self.all_parameters[trackName]["tick_spacing"] = value
        self._display_depth_track(self.depth_curve_name)

    def updateLineSpacing(self, trackName, value):
        self.track_parameters[trackName]["line_spacing"] = value
        self.all_parameters[trackName]["line_spacing"] = value
        self._display_depth_track(self.depth_curve_name)

    def onDisplayNameChanged(self, text):
        col = self.sender().property("column_name")
        track = self.sender().property("track_name")  # Get the trackName
        self.updateCurveDisplayName(track, col, text)

    def updateCurveDisplayName(self, trackName, col, text):
        print(f"Updating display name for track: {trackName}, column: {col} to {text}")  # Debug line
        if trackName in self.all_parameters and 'curves' in self.all_parameters[trackName]:
            self.all_parameters[trackName]['curves'][col]['display_name'] = text
        if trackName in self.track_plot_items and 'textItems' in self.track_plot_items[trackName]:
            if col in self.track_plot_items[trackName]['textItems']:
                # Retrieve the TextItem
                textItem = self.track_plot_items[trackName]['textItems'][col]

                # Update the color of the TextItem
                textItem.setText(text)

    def updateLineStyle(self, column, trackName, selectedStyle):
        # Map the textual representation to the Qt enumeration (or whatever representation you use internally)
        internal_style = self.pen_styles.get(selectedStyle, Qt.PenStyle.SolidLine)

        # Update the all_parameters dictionary
        if trackName in self.all_parameters and 'curves' in self.all_parameters[trackName]:
            self.all_parameters[trackName]['curves'][column]["line_style"] = selectedStyle

        # Update Pen-style
        if 'logs' in self.track_plot_items[trackName] and column in self.track_plot_items[trackName]['logs']:
            logItem = self.track_plot_items[trackName]['logs'][column]
            pen = logItem.opts['pen']
            pen.setStyle(internal_style)
            logItem.setPen(pen)

    def updateFillColorPalette(self, column, trackName, selectedPalette):
        # Update the all_parameters dictionary
        if trackName in self.all_parameters and 'curves' in self.all_parameters[trackName]:
            self.all_parameters[trackName]['curves'][column]["fill_color_palette"] = selectedPalette

            # Check and remove the existing logItem if it exists
            if 'logs' in self.track_plot_items[trackName] and column in self.track_plot_items[trackName]['logs']:
                old_logItem = self.track_plot_items[trackName]['logs'][column]
                plotItem = self.track_plot_items[trackName]['plot']
                plotItem.removeItem(old_logItem)

            # Replot the curve with the updated filling side
            self.plot_curve(trackName, column)

    def updateFillingSide(self, column, trackName, selectedSide):
        # Update the all_parameters dictionary
        if trackName in self.all_parameters and 'curves' in self.all_parameters[trackName]:
            self.all_parameters[trackName]['curves'][column]["filling_side"] = selectedSide

            # Check and remove the existing logItem if it exists
            if 'logs' in self.track_plot_items[trackName] and column in self.track_plot_items[trackName]['logs']:
                old_logItem = self.track_plot_items[trackName]['logs'][column]
                plotItem = self.track_plot_items[trackName]['plot']
                plotItem.removeItem(old_logItem)

            # Replot the curve with the updated filling side
            self.plot_curve(trackName, column)

    def updateMinValue(self, column, trackName, value):
        if trackName in self.all_parameters and 'curves' in self.all_parameters[trackName]:
            self.all_parameters[trackName]['curves'][column]["min_val"] = value
        plotItem = self.track_plot_items[trackName]["plot"]
        scale = self.all_parameters[trackName]["scale"]
        if scale == "normal":
            plotItem.setXRange(value, self.all_parameters[trackName]['curves'][column].get("max_val"), padding=0)
        elif scale == "log scale":
            if self.all_parameters[trackName]['curves'][column].get("max_val") > 0 and value > 0:
                plotItem.setXRange(math.log10(value),
                                   math.log10(self.all_parameters[trackName]['curves'][column].get("max_val")),
                                   padding=0)

    def updateMaxValue(self, column, trackName, value):
        if trackName in self.all_parameters and 'curves' in self.all_parameters[trackName]:
            self.all_parameters[trackName]['curves'][column]["max_val"] = value
        plotItem = self.track_plot_items[trackName]["plot"]
        scale = self.all_parameters[trackName]["scale"]
        if scale == "normal":
            plotItem.setXRange(self.all_parameters[trackName]['curves'][column].get("min_val"), value, padding=0)
        elif scale == "log scale":
            if self.all_parameters[trackName]['curves'][column].get("min_val") > 0 and value > 0:
                plotItem.setXRange(math.log10(self.all_parameters[trackName]['curves'][column].get("min_val")),
                                   math.log10(value), padding=0)

    def updateReverse(self, column, trackName, isChecked):
        if trackName in self.all_parameters and 'curves' in self.all_parameters[trackName]:
            self.all_parameters[trackName]['curves'][column]["reverse"] = isChecked

        plotItem = self.track_plot_items[trackName]["plot"]
        if isChecked:
            plotItem.invertX(True)
        else:
            plotItem.invertX(False)

    def updateLeftCutOff(self, column, trackName, value):
        if trackName in self.all_parameters and 'curves' in self.all_parameters[trackName]:
            self.all_parameters[trackName]['curves'][column]["left_cut_off"] = value
        lineItem = self.track_plot_items[trackName]['left_cut_off'][column]
        if self.all_parameters[trackName]["scale"] == "log scale":
            if value == 0:
                return
            value = np.log10(value)
        lineItem.setPos(value)

    def updateRightCutOff(self, column, trackName, value):
        if trackName in self.all_parameters and 'curves' in self.all_parameters[trackName]:
            self.all_parameters[trackName]['curves'][column]["right_cut_off"] = value
        lineItem = self.track_plot_items[trackName]['right_cut_off'][column]
        if self.all_parameters[trackName]["scale"] == "log scale":
            if value == 0:
                return
            value = np.log10(value)
        lineItem.setPos(value)

    def generate_argb(self, color, alpha):
        return f"#{alpha:02X}{color[1:]}"

    def removeCurveTab(self, col, trackName):
        # 1. Find the tab corresponding to the trackName
        trackTabIndex = -1
        for i in range(self.tracksTabWidget.count()):
            if self.tracksTabWidget.tabText(i) == trackName:
                trackTabIndex = i
                break

        if trackTabIndex == -1:
            return  # Didn't find the track

        trackTab = self.tracksTabWidget.widget(trackTabIndex)

        # 2. Find the QTabWidget within the trackTab
        curvesTabWidget = None
        trackLayout = trackTab.layout()
        if trackLayout:  # Check to ensure the layout exists
            for i in range(trackLayout.count()):
                widget = trackLayout.itemAt(i).widget()
                if isinstance(widget, QTabWidget):
                    curvesTabWidget = widget
                    break

        if not curvesTabWidget:
            return  # Didn't find the curves tab widget

        # 3. Find the tab by its object name and remove it
        widget_with_name_col = curvesTabWidget.findChild(QWidget, col)
        if widget_with_name_col:
            curveTabIndex = curvesTabWidget.indexOf(widget_with_name_col)

            if curveTabIndex != -1:
                curvesTabWidget.removeTab(curveTabIndex)
                widget_with_name_col.deleteLater()

        # Now, remove the curve parameters from the current track as well
        if trackName in self.all_parameters and 'curves' in self.all_parameters[trackName] and col in \
                self.all_parameters[trackName]['curves']:
            del self.all_parameters[trackName]['curves'][col]

        # Remove the curve from GraphWidget
        if trackName in self.track_plot_items and 'logs' in self.track_plot_items[trackName] and col in \
                self.track_plot_items[trackName]['logs']:
            textItem = self.track_plot_items[trackName]['textItems'].pop(col)
            logItem = self.track_plot_items[trackName]['logs'].pop(col)

        # Get the plotItem for the track
        plotItem = self.track_plot_items[trackName]['plot']
        # Retrieve the curveItem for this track
        curveItem = self.track_plot_items[trackName]['curve']

        plotItem.removeItem(logItem)
        curveItem.removeItem(textItem)

    def openColorDialog(self, curveName, attribute, colorButton, alphaSlider, context="color"):
        # Open color dialog only if the context is 'color'
        if context == "color":
            color = QColorDialog.getColor()
            if color.isValid():
                hexColor = color.name()
                alpha = alphaSlider.value()
                colorButton.setText(hexColor)
            else:
                # If the color dialog was canceled, exit the function
                return
        else:
            # If context is not 'color', retrieve the existing color from the button
            hexColor = colorButton.text()
            alpha = alphaSlider.value()

        # Set the RGBA color for the button
        colorButton.setStyleSheet(
            f"background-color: rgba({int(hexColor[1:3], 16)}, {int(hexColor[3:5], 16)}, {int(hexColor[5:7], 16)}, {alpha}); color: #FFFFFF;")

        # Update the curve_parameters dictionary
        if curveName not in self.curve_parameters:
            self.curve_parameters[curveName] = {}

        self.curve_parameters[curveName][f"{attribute}_color"] = hexColor
        self.curve_parameters[curveName][f"{attribute}_alpha"] = alpha

        # Get the column and track names from the associated QLineEdit
        column = colorButton.parent().findChild(QLineEdit).property("column_name")
        trackName = colorButton.parent().findChild(QLineEdit).property("track_name")

        # Update the all_parameters dictionary
        if trackName in self.all_parameters and 'curves' in self.all_parameters[trackName]:
            self.all_parameters[trackName]['curves'][column][f"{attribute}_color"] = hexColor
            self.all_parameters[trackName]['curves'][column][f"{attribute}_alpha"] = alpha

        if trackName == "DEPTH":
            self._display_depth_track(column)
            return

        # Update Pen-style
        color = pg.mkColor(hexColor)
        color.setAlpha(alpha)
        if attribute == 'line':
            if 'logs' in self.track_plot_items[trackName] and column in self.track_plot_items[trackName]['logs']:
                logItem = self.track_plot_items[trackName]['logs'][column]
                pen = logItem.opts['pen']
                pen.setColor(color)
                logItem.setPen(pen)

            if trackName in self.track_plot_items and 'textItems' in self.track_plot_items[trackName]:
                if column in self.track_plot_items[trackName]['textItems']:
                    # Retrieve the TextItem
                    textItem = self.track_plot_items[trackName]['textItems'][column]

                    # Update the color of the TextItem
                    textItem.setColor(color)
                else:
                    print(f"Text item for column {column} not found in track {trackName}")
            else:
                print(f"Track name {trackName} not found or no text items available")

        if attribute == 'left_cut_off':
            if 'left_cut_off' in self.track_plot_items[trackName] and column in self.track_plot_items[trackName][
                "left_cut_off"]:
                lineItem = self.track_plot_items[trackName]["left_cut_off"][column]
                pen = lineItem.pen
                pen.setColor(color)
                lineItem.setPen(pen)

        if attribute == 'right_cut_off':
            if 'right_cut_off' in self.track_plot_items[trackName] and column in self.track_plot_items[trackName][
                "right_cut_off"]:
                lineItem = self.track_plot_items[trackName]["right_cut_off"][column]
                pen = lineItem.pen
                pen.setColor(color)
                lineItem.setPen(pen)

    def updateButtonTransparency(self, colorButton, alphaSlider):
        # Extract RGB from the current color
        currentColor = colorButton.palette().button().color()
        red = currentColor.red()
        green = currentColor.green()
        blue = currentColor.blue()
        alpha = alphaSlider.value()
        # Update the button's background color with the new transparency
        colorButton.setStyleSheet(f"background-color: rgba({red}, {green}, {blue}, {alpha}); color: #FFFFFF;")

    def loadLAS(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open LAS File", "", "LAS Files (*.las);;All Files (*)")

        if filePath:
            self.las_filename = ".".join(filePath.split("/")[-1].split(".")[:-1])
            self.currentFileNameTextBox.setText(f"Current Well: {self.las_filename}")
            las = lasio.read(filePath)
            df = las.df().reset_index()  # Reset index to make depth a regular column
            self.depth_column_name = df.columns[0]

            # Block signals while updating the table
            self.table.blockSignals(True)
            self.table.clear()

            self.table.setRowCount(df.shape[0])
            self.table.setColumnCount(df.shape[1])

            for i, (index, row) in enumerate(df.iterrows()):
                for j, value in enumerate(row):
                    self.table.setItem(i, j, QTableWidgetItem(str(value)))

            self.table.setHorizontalHeaderLabels(df.columns.tolist())
            self.curve_exist = df.columns.tolist()
            self.las = las
            self.df = df

            # Re-enable signals after updating the table
            self.table.blockSignals(False)

            # Clear and reset the Track tab, Table tab, and plotWidget
            self.removeAllTracks()
            self.topsTable.clear()

        # Connect the signal to the slot function
        self.table.itemChanged.connect(self.on_table_item_changed)

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

    def load_xml(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load XML Template", "", "XML Files (*.xml);;All Files (*)")
        if file_name:
            with open(file_name, 'r') as file:
                self.xml_data = file.read()
            self.plotData()

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
                excel_file_name = ".".join(
                    filePath.split("/")[-1].split(".")[:-1])  # Extract file name without extension

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

    def loadTops(self):
        """
        Load tops data from a CSV file and populate the topsTable.

        The function opens a file dialog to select a CSV file, reads the contents,
        and populates the topsTable with data corresponding to the UWI value of the
        currently loaded LAS file. It also adds a 'Color Palette' column with a color button.
        """
        # Check if LAS data is loaded
        if not hasattr(self, 'las') or not self.las:
            QMessageBox.warning(self, "Load LAS File", "Please load a LAS file first.")
            return

        self.topfilePath, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")

        if self.topfilePath:
            self.topsTable.blockSignals(True)

            with open(self.topfilePath, 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                headers = next(reader)

                # Determine the index of the UWI column
                uwi_index = headers.index('UWI')

                self.topsTable.setRowCount(0)
                self.topsTable.setColumnCount(len(headers) + 1)
                self.topsTable.setHorizontalHeaderLabels(headers + ["Color Palette"])
                for row_data in reader:
                    if row_data[uwi_index] == self.las.well.UWI.value:
                        row = self.topsTable.rowCount()
                        self.topsTable.insertRow(row)
                        for column, data in enumerate(row_data):
                            item = QTableWidgetItem(data)
                            self.topsTable.setItem(row, column, item)
                        colorValue = row_data[-1]
                        colorButton = self.createColorButton()
                        colorButton.setStyleSheet(f"background-color: {colorValue};")
                        self.topsTable.setCellWidget(row, len(row_data), colorButton)

            self.topsTable.blockSignals(False)

            # Draw the tops for the first time using the current view range
            self.initialDrawTops()
            self.topsTable.itemChanged.connect(self.onTopTableChanged)

    def initialDrawTops(self):
        self.removeTops()
        for trackName, trackItems in self.track_plot_items.items():
            plotItem = trackItems['plot']
            viewRange = plotItem.getViewBox().viewRange()
            x_min, x_max = viewRange[0]
            topsData = self.extractTopsData()
            self.drawTops(topsData, x_min, x_max, plotItem)

    def createColorButton(self):
        colorButton = QPushButton('Color')
        colorButton.clicked.connect(self.chooseColor)
        return colorButton

    def chooseColor(self):
        color = QColorDialog.getColor()
        if color.isValid():
            sender = self.sender()
            row = self.topsTable.indexAt(sender.pos()).row()

            # Find the column index with the header 'Color'
            colorColumnIndex = self.findColumnIndex("Color")

            if colorColumnIndex is not None:
                colorName = color.name()

                # Update the cell with the color name
                self.topsTable.setItem(row, colorColumnIndex, QTableWidgetItem(colorName))

                # Redraw the table or cell (if needed)
                self.topsTable.update()

                # Optional: Update button color as a visual indicator
                sender.setStyleSheet(f"background-color: {colorName};")

    def saveTops(self):
        """
        Save the modified tops data back to the original CSV file.

        The function extracts modified rows from the table, groups them by UWI,
        and then updates these groups in the original data read from the CSV file.
        It skips the 'Color Palette' column and handles adding new UWI groups if necessary.

        The updated data is then saved back to the CSV file.
        """
        try:
            filePath = self.topfilePath  # This should be set when you load the file

            if not filePath:
                raise Exception("File path not set. Unable to save data.")

            # Find the index of the 'Color Palette' column
            colorPaletteColumnIndex = self.findColumnIndex("Color Palette")

            # Extract modified rows from the table, excluding 'Color Palette' column
            modifiedData = []
            uwiIndex = self.findColumnIndex("UWI")
            for rowIndex in range(self.topsTable.rowCount()):
                rowData = []
                for colIndex in range(self.topsTable.columnCount()):
                    if colIndex != colorPaletteColumnIndex:
                        item = self.topsTable.item(rowIndex, colIndex)
                        rowData.append(item.text() if item else '')
                modifiedData.append(rowData)

            # Group modified data by UWI
            modifiedDataByUWI = {}
            for row in modifiedData:
                uwiNumber = row[uwiIndex]
                if uwiNumber not in modifiedDataByUWI:
                    modifiedDataByUWI[uwiNumber] = []
                modifiedDataByUWI[uwiNumber].append(row)

            # Read original file and store data
            with open(filePath, 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                originalData = list(reader)

            # Replace relevant UWI group in original data
            updatedData = []
            skipUWI = None  # UWI to skip as it's already processed
            for row in originalData:
                if row[uwiIndex] == skipUWI:
                    continue  # Skip remaining rows of the same UWI group

                if row[uwiIndex] in modifiedDataByUWI:
                    updatedData.extend(modifiedDataByUWI[row[uwiIndex]])  # Replace group
                    skipUWI = row[uwiIndex]  # Set this UWI to be skipped in subsequent rows
                else:
                    updatedData.append(row)  # Keep other UWIs unchanged

            # Save the updated data back to the file
            with open(filePath, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(updatedData)
                QMessageBox.information(self, "Save Successful", "The tops data has been successfully saved.")


        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"An error occurred while saving tops: {e}")

    def addTop(self):
        currentRow = self.topsTable.currentRow() + 1
        newRowIndex = currentRow if currentRow >= 0 else self.topsTable.rowCount()

        # Find the column index for 'UWI' and 'Chg Date'
        uwiColumnIndex = self.findColumnIndex("UWI")
        chgDateColumnIndex = self.findColumnIndex("Chg Date")

        self.topsTable.insertRow(newRowIndex)

        # Find the first non-None UWI value in the column
        uwiValue = ''
        for rowIndex in range(self.topsTable.rowCount()):
            item = self.topsTable.item(rowIndex, uwiColumnIndex)
            if item is not None and item.text():
                uwiValue = item.text()
                break

        # Set the UWI value in the new row
        if uwiColumnIndex is not None:
            self.topsTable.setItem(newRowIndex, uwiColumnIndex, QTableWidgetItem(uwiValue))

        # Add a ColorButton in the new row
        colorButtonColumnIndex = self.topsTable.columnCount() - 1  # Assuming the last column
        colorButton = self.createColorButton()
        self.topsTable.setCellWidget(newRowIndex, colorButtonColumnIndex, colorButton)

        # Add current timestamp in 'Chg Date' column
        if chgDateColumnIndex is not None:
            currentDate = datetime.now().strftime("%m/%d/%Y")
            self.topsTable.setItem(newRowIndex, chgDateColumnIndex, QTableWidgetItem(currentDate))

    def deleteTop(self):
        currentRow = self.topsTable.currentRow()

        # Check if a row is selected
        if currentRow >= 0:
            self.topsTable.removeRow(currentRow)

            # After updating the topsTable
            topsData = self.extractTopsData()

            # Assuming each track has its plotItem stored in track_plot_items
            for trackName, trackItems in self.track_plot_items.items():
                plotItem = trackItems['plot']
                viewRange = plotItem.getViewBox().viewRange()
                x_min, x_max = viewRange[0]

                # Call drawTops with the updated parameters
                self.drawTops(topsData, x_min, x_max, plotItem)

        else:
            # Optional: Show a message that no row is selected
            pass

    def removeTops(self):
        for trackName, trackItems in self.track_plot_items.items():
            plotItem = trackItems['plot']

            # Clear existing line and fill items
            if hasattr(plotItem, 'topsLineItems'):
                for item in plotItem.topsLineItems:
                    plotItem.removeItem(item)
            if hasattr(plotItem, 'topsFillItems'):
                for item in plotItem.topsFillItems:
                    plotItem.removeItem(item)

            plotItem.topsLineItems = []
            plotItem.topsFillItems = []

    def onTopTableChanged(self, item):
        # Check if the changed item is in the 'Color' column
        colorColumnIndex = self.findColumnIndex("Color")
        colorButtonIndex = self.findColumnIndex("Color Palette")

        if item.column() == colorColumnIndex:
            colorValue = item.text()
            colorButton = self.topsTable.cellWidget(item.row(), colorButtonIndex)
            if colorButton:
                colorButton.setStyleSheet(f"background-color: {colorValue};")

        # After updating the topsTable
        topsData = self.extractTopsData()

        # Assuming each track has its plotItem stored in track_plot_items
        for trackName, trackItems in self.track_plot_items.items():
            plotItem = trackItems['plot']
            viewRange = plotItem.getViewBox().viewRange()
            x_min, x_max = viewRange[0]

            # Call drawTops with the updated parameters
            self.drawTops(topsData, x_min, x_max, plotItem)

    def onRangeChanged(self, plotItem, viewRange):
        x_min, x_max = viewRange[0]  # Extract the x range
        # Assuming topsData is stored in the plotItem
        topsData = self.extractTopsData()
        if topsData:
            self.drawTops(topsData, x_min, x_max, plotItem)

    def findColumnIndex(self, columnName):
        for columnIndex in range(self.topsTable.columnCount()):
            if self.topsTable.horizontalHeaderItem(columnIndex).text() == columnName:
                return columnIndex
        return -1

    def findColumnIndexWithSubstring(self, substring):
        for columnIndex in range(self.topsTable.columnCount()):
            headerText = self.topsTable.horizontalHeaderItem(columnIndex).text()
            if substring in headerText:
                return columnIndex
        return -1  # Return -1 if the substring is not found in any header

    def extractTopsData(self):
        topsData = []
        mdIndex = self.findColumnIndex("MD")
        formationIndex = self.findColumnIndexWithSubstring("(Fm Name)")
        colorIndex = self.findColumnIndex("Color")

        for rowIndex in range(self.topsTable.rowCount()):
            item = self.topsTable.item(rowIndex, mdIndex)
            mdStr = item.text() if item is not None else '0'
            md = float(mdStr.replace(',', '')) if mdStr else 0.0

            formationItem = self.topsTable.item(rowIndex, formationIndex)
            formation = formationItem.text() if formationItem is not None else ''

            colorItem = self.topsTable.item(rowIndex, colorIndex)
            color = colorItem.text() if colorItem is not None else ''

            topsData.append((md, formation, color))

        return topsData

    def drawTops(self, topsData, x_min, x_max, plotItem):
        # for trackName, trackItems in self.track_plot_items.items():
        #     plotItem = trackItems['plot']
        x_min = pow(10, x_min) if plotItem.getAxis('bottom').logMode else x_min
        x_max = pow(10, x_max) if plotItem.getAxis('bottom').logMode else x_max

        # Clear existing line and fill items
        if hasattr(plotItem, 'topsLineItems'):
            for item in plotItem.topsLineItems:
                plotItem.removeItem(item)
        if hasattr(plotItem, 'topsFillItems'):
            for item in plotItem.topsFillItems:
                plotItem.removeItem(item)

        plotItem.topsLineItems = []
        plotItem.topsFillItems = []
        prevMd = None
        prevLine = None
        prevColor = None

        for md, formation, color in topsData:
            # Check if color is a valid color string and default to black if not
            if not color.startswith('#'):
                color = pg.mkColor('#000000')  # Default to black
                brushcolor = pg.mkColor('#000000')
                brushcolor.setAlpha(0)
            else:
                brushcolor = pg.mkColor(color)
                brushcolor.setAlpha(100)

            # Create a horizontal line using PlotDataItem
            line = pg.PlotDataItem(
                x=[x_min, x_max], y=[md, md],
                pen=pg.mkPen(color))
            plotItem.addItem(line)
            plotItem.topsLineItems.append(line)

            # Add formation name
            textItem = pg.TextItem(formation, anchor=(0.5, 0.5))
            plotItem.addItem(textItem)
            textItem.setPos(0.5, md)  # Adjust position as needed

            # Fill color between lines
            if prevMd is not None and prevLine is not None:
                fill = pg.FillBetweenItem(curve1=line, curve2=prevLine, brush=pg.mkBrush(prevColor))
                plotItem.addItem(fill)
                plotItem.topsFillItems.append(fill)

            prevMd = md
            prevLine = line
            prevColor = brushcolor
        # Check if there's at least one line (and thus a valid prevLine)
        if prevLine is not None:
            # Get the last data point's y-value
            viewRange = plotItem.getViewBox().viewRange()
            _, y_max = viewRange[1]
            # Create an additional line at the last data point
            brushcolor = pg.mkColor('#000000')
            brushcolor.setAlpha(0)
            lastLine = pg.PlotDataItem(
                x=[x_min, x_max], y=[y_max, y_max], pen=pg.mkPen(brushcolor))
            plotItem.addItem(lastLine)

            # Fill between the last actual line and this new line
            finalFill = pg.FillBetweenItem(
                curve1=prevLine, curve2=lastLine, brush=pg.mkBrush(prevColor))
            plotItem.addItem(finalFill)
            plotItem.topsFillItems.append(finalFill)

    def generate_palette_preview(self, palette_name):
        """Generate a QPixmap from a matplotlib colormap."""
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.vstack((gradient, gradient))
        fig, ax = plt.subplots(figsize=(0.6, 0.25))
        fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        ax.set_title(palette_name)
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(palette_name))
        ax.set_axis_off()
        fig.canvas.draw()

        # Convert fig to QPixmap
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        pixmap = QImage(img.data, img.shape[1], img.shape[0], QImage.Format.Format_RGB888)
        return QPixmap.fromImage(pixmap)

    def plot_track(self, trackName, index):

        if trackName in self.track_columns:
            params = self.all_parameters.get(trackName, {})

            track_width = params.get('track_width', 1)
            # Calculate the total width from the self.all_parameters
            borderPen = pg.mkPen(width=2)
            col = index

            # Create a subplot for track title
            titleItem = self.mainGraphicsLayout.addPlot(row=0, col=col)
            titleItem.setMaximumHeight(40)  # Set maximum height

            titleItem.hideAxis('bottom')
            titleItem.showAxis('right')
            titleItem.showAxis('top')
            if col > 0:
                titleItem.hideAxis('left')
            titleItem.getAxis('left').setTicks([])  # Hide left axis ticks
            titleItem.getAxis('right').setTicks([])  # Hide right axis ticks
            titleItem.getAxis('top').setTicks([])  # Hide top axis ticks
            titleItem.getAxis('bottom').setTicks([])  # Hide bottom axis ticks
            titleItem.getAxis('left').setPen(borderPen)
            titleItem.getAxis('right').setPen(borderPen)
            titleItem.getAxis('top').setPen(borderPen)
            titleItem.getAxis('left').setStyle(showValues=False)
            titleItem.getAxis('right').setStyle(showValues=False)
            titleItem.getAxis('top').setStyle(showValues=False)
            titleItem.getAxis('bottom').setStyle(showValues=False)

            titleText = AnchoredTextItem(text=params.get("track_display_name", trackName), anchor=(0.5, 0.5))
            titleItem.addItem(titleText)
            titleText.setPos(0.5, 0.5)  # Top-center position
            titleItem.getViewBox().setContentsMargins(0, 0, 0, 0)

            titleItem.update()

            # Add the new curveItem row between title and plot
            curveItem = self.mainGraphicsLayout.addPlot(row=1, col=col)
            curveItem.setMaximumHeight(60)  # Set maximum height

            # Configure the curveItem as per your requirements, e.g., hide axes, set background color, etc.
            curveItem.showAxis('right')
            curveItem.showAxis('top')
            if col > 0:
                curveItem.hideAxis('left')
            curveItem.hideAxis('bottom')
            curveItem.getAxis('left').setTicks([])  # Hide left axis ticks
            curveItem.getAxis('right').setTicks([])  # Hide right axis ticks
            curveItem.getAxis('top').setTicks([])  # Hide top axis ticks
            curveItem.getAxis('bottom').setTicks([])  # Hide bottom axis ticks
            curveItem.getAxis('left').setPen(borderPen)
            curveItem.getAxis('right').setPen(borderPen)
            curveItem.getAxis('top').setPen(borderPen)
            curveItem.getAxis('left').setStyle(showValues=False)
            curveItem.getAxis('right').setStyle(showValues=False)
            curveItem.getAxis('top').setStyle(showValues=False)
            curveItem.getAxis('bottom').setStyle(showValues=False)
            curveItem.update()

            # Create the main track
            plotItem = self.mainGraphicsLayout.addPlot(row=2, col=col)
            plotItem.setXRange(0, track_width)

            # Borders for the Track
            plotItem.getAxis('left').setPen(borderPen)
            plotItem.getAxis('bottom').setPen(borderPen)
            plotItem.getAxis('right').setPen(borderPen)
            plotItem.getAxis('right').setStyle(showValues=False)  # This will ensure the right axis doesn't show numbers
            plotItem.getAxis('top').setPen(borderPen)
            plotItem.getAxis('top').setStyle(showValues=False)  # This will ensure the right axis doesn't show numbers
            if col > 0:
                plotItem.hideAxis('left')
            plotItem.showAxis('right')
            plotItem.showAxis('top')
            plotItem.getAxis('left').setStyle(showValues=False)
            plotItem.getAxis('right').setStyle(showValues=False)
            plotItem.getAxis('top').setStyle(showValues=False)
            plotItem.getAxis('bottom').setStyle(showValues=False)

            # Set the width (Y-axis range since we swapped)
            plotItem.getViewBox().setContentsMargins(0, 0, 0, 0)

            plotItem.update()

            # Store the plotItem for further reference or update if exists
            if trackName in self.track_plot_items:
                # Update existing plot with new data
                self.track_plot_items[trackName].setData(...)  # <- replace with actual data update method
            else:
                self.track_plot_items[trackName] = {
                    'title': titleItem,
                    'curve': curveItem,
                    'plot': plotItem
                }

            if col > 0:
                firstTrackKey = list(self.track_plot_items.keys())[0]
                firstPlotItem = self.track_plot_items[firstTrackKey][
                    'plot']  # Access the 'plot' item from the first track
                plotItem.setYLink(firstPlotItem)  # Link Y-axis with the first plot item

    def removePlotForTrack(self, trackName, currentIndex):
        if trackName in self.track_plot_items:
            # Retrieve both title and plot items
            titleItem = self.track_plot_items[trackName]['title']
            curveItem = self.track_plot_items[trackName]['curve']
            plotItem = self.track_plot_items[trackName]['plot']

            # Remove both items from the main graphics layout
            self.mainGraphicsLayout.removeItem(titleItem)
            self.mainGraphicsLayout.removeItem(curveItem)
            self.mainGraphicsLayout.removeItem(plotItem)

            # Remove the entry from the track plot items dictionary
            del self.track_plot_items[trackName]
            # Shift remaining tracks to the left
            self.recreateLayout(trackName)

            # If the removed track was the first one, update the left axis of the new first track
            if currentIndex == 0 and len(self.track_plot_items) > 0:
                newFirstTrackName = list(self.track_plot_items.keys())[0]
                newFirstPlotItem = self.track_plot_items[newFirstTrackName]['plot']
                newFirstCurveItem = self.track_plot_items[newFirstTrackName]['curve']
                newFirstTitleItem = self.track_plot_items[newFirstTrackName]['title']
                newFirstPlotItem.showAxis('left')
                newFirstCurveItem.showAxis('left')
                newFirstTitleItem.showAxis('left')

        else:
            # Optionally handle the case where the track name is not found
            print(f"Track name '{trackName}' not found in track plot items.")

    def recreateLayout(self, removedTrackName):
        # Store the remaining tracks temporarily
        remainingTracks = {k: v for k, v in self.track_plot_items.items() if k != removedTrackName}

        # Clear all items in the layout
        self.mainGraphicsLayout.clear()

        # Re-add the remaining items in the new order
        for index, (trackName, plotItems) in enumerate(remainingTracks.items()):
            # Add each component of the track back to the layout in the new position
            self.mainGraphicsLayout.addItem(plotItems['title'], row=0, col=index)
            self.mainGraphicsLayout.addItem(plotItems['curve'], row=1, col=index)
            self.mainGraphicsLayout.addItem(plotItems['plot'], row=2, col=index)

        # Update and repaint the plotWidget (GraphicsView)
        self.plotWidget.update()
        self.plotWidget.repaint()

        # Optionally, reset the central item of the plotWidget
        self.plotWidget.setCentralItem(None)
        self.plotWidget.setCentralItem(self.mainGraphicsLayout)

        # Update the track_plot_items dictionary to reflect the new layout
        self.track_plot_items = remainingTracks

    def swapTracks(self, index1, index2):
        # Swap the tracks in the mainGraphicsLayout
        trackName1 = self.tracksTabWidget.tabText(index1)
        trackName2 = self.tracksTabWidget.tabText(index2)

        if trackName1 in self.track_plot_items and trackName2 in self.track_plot_items:
            plotItem1 = self.track_plot_items[trackName1]['plot']
            curveItem1 = self.track_plot_items[trackName1]['curve']
            titleItem1 = self.track_plot_items[trackName1]['title']
            plotItem2 = self.track_plot_items[trackName2]['plot']
            curveItem2 = self.track_plot_items[trackName2]['curve']
            titleItem2 = self.track_plot_items[trackName2]['title']

            # Remove both items
            self.mainGraphicsLayout.removeItem(plotItem1)
            self.mainGraphicsLayout.removeItem(plotItem2)
            self.mainGraphicsLayout.removeItem(titleItem1)
            self.mainGraphicsLayout.removeItem(titleItem2)
            self.mainGraphicsLayout.removeItem(curveItem1)
            self.mainGraphicsLayout.removeItem(curveItem2)

            # Reinsert in swapped positions
            self.mainGraphicsLayout.addItem(titleItem1, row=0, col=index2)
            self.mainGraphicsLayout.addItem(titleItem2, row=0, col=index1)
            self.mainGraphicsLayout.addItem(curveItem1, row=1, col=index2)
            self.mainGraphicsLayout.addItem(curveItem2, row=1, col=index1)
            self.mainGraphicsLayout.addItem(plotItem1, row=2, col=index2)
            self.mainGraphicsLayout.addItem(plotItem2, row=2, col=index1)

            # Ensure left axis visibility is correct after swapping
            if index1 == 0 or index2 == 0:  # If one of the tracks is the first track
                for i, plotItem in enumerate([plotItem1, plotItem2]):
                    titleItem = [titleItem1, titleItem2][i]  # Get the corresponding title item
                    curveItem = [curveItem1, curveItem2][i]
                    if self.mainGraphicsLayout.getItem(2, 0) == plotItem:  # If it's the first track now
                        plotItem.showAxis('left')
                        titleItem.showAxis('left')
                        curveItem.showAxis('left')
                    else:
                        # If it's not the first track anymore, hide the left axis
                        plotItem.hideAxis('left')
                        titleItem.hideAxis('left')
                        curveItem.hideAxis('left')

        else:
            print("Track names not found in track plot items.")

    def updateTrackWidths(self):
        if self.total_TrackWidth == 0:
            # Handle the case where there are no tracks
            # For example, you might want to reset the widths to a default state or do nothing
            return

        # Calculate the total width from the self.all_parameters
        self.total_TrackWidth = sum(param.get('track_width', 1) for param in self.all_parameters.values())
        for trackName, plotItems in self.track_plot_items.items():
            track_width = self.all_parameters.get(trackName, {}).get('track_width', 1)
            width_ratio = track_width / self.total_TrackWidth

            # Calculate the new width based on the available space and the width ratio
            new_width = self.plotWidget.width() * width_ratio * 0.985

            # Set the new width for the title and plot items
            plotItems['title'].setFixedWidth(new_width)
            plotItems['curve'].setFixedWidth(new_width)
            plotItems['plot'].setFixedWidth(new_width)

        # Update the layout
        self.plotWidget.update()
        self.plotWidget.scene().update()

    def updateTitle(self, trackName, newTitle):
        if trackName in self.track_plot_items:
            titleItem = self.track_plot_items[trackName]['title']
            for item in titleItem.items:
                if isinstance(item, AnchoredTextItem):  # Check if the item is our title text
                    item.setText(newTitle)
                    titleItem.update()
        else:
            print(f"Track name '{trackName}' not found in track plot items.")

    def plot_curve(self, trackName, col):
        if trackName not in self.track_plot_items:
            print(f"Track {trackName} not found!")
            return

        # Handle the special DEPTH track
        if trackName == "DEPTH":
            self.depth_curve_name = col
            self._display_depth_track(self.depth_curve_name)
            return

        self.padding = 0
        curve_data = self.get_column_data(col)
        curve_data = np.array(curve_data, dtype=float)
        curve_min = np.nanmin(curve_data)
        curve_max = np.nanmax(curve_data)

        # Find indices of the first and last non-NaN values
        non_nan_indices = np.where(~np.isnan(curve_data))[0]
        first_non_nan = non_nan_indices[0]
        last_non_nan = non_nan_indices[-1]
        curve_data = curve_data[first_non_nan:last_non_nan + 1]

        # Retrieve depth data (assuming 'DEPT' is your depth column)
        depth_data = self.get_column_data(self.depth_curve_name)
        depth_data = np.array(depth_data, dtype=float)

        if trackName == "GAMMA RAY":
            # Set the first and last values of curve_data to 0
            curve_data[0] = 0
            curve_data[-1] = 0

        depth_data = depth_data[first_non_nan:last_non_nan + 1]

        min_depth = np.nanmin(depth_data)
        max_depth = np.nanmax(depth_data)

        # Retrieve the visualization parameters for this track
        trackparams = self.all_parameters[trackName]
        params = self.all_parameters[trackName]['curves'][col]
        # Get the plotItem for the track
        plotItem = self.track_plot_items[trackName]['plot']
        # Retrieve the curveItem for this track
        curveItem = self.track_plot_items[trackName]['curve']

        # Create a linked AxisItem
        linkedAxis = pg.AxisItem(orientation='bottom', text=params['display_name'])
        linkedAxis.linkToView(plotItem.getViewBox())
        if trackparams["scale"] == "log scale":
            linkedAxis.setLogMode(x=True, y=False)

        # Add the linked Axis to the curveItem
        curveItem.layout.addItem(linkedAxis, 2, 1)  # Add it to the bottom of the curveItem

        # Create and add the TextItem
        curveName = f"{col}"
        textItem = pg.TextItem(text=curveName, anchor=(0.5, 0.5), color=params["line_color"])

        # Add the TextItem to the curveItem
        curveItem.addItem(textItem)
        # Check if 'textItems' container exists for the 'curve', if not, initialize it
        if 'textItems' not in self.track_plot_items[trackName]:
            self.track_plot_items[trackName]['textItems'] = {}
        # Store the TextItem in the track_plot_items dictionary
        self.track_plot_items[trackName]['textItems'][col] = textItem

        # Define a vertical offset between each TextItem
        vertical_offset = 1
        # Use the view box margins to determine the initial Y position
        initial_y_position = 20  # Start below the bottom margin
        # Calculate the number of existing TextItems in curveItem
        num_text_items = sum(1 for item in curveItem.items if isinstance(item, pg.TextItem))

        # Set a static position for the TextItem
        # Assuming the size of curveItem is known and doesn't change drastically
        static_x_position = curveItem.width() / 2  # Center horizontally
        static_y_position = initial_y_position + (vertical_offset * num_text_items)  # A small offset from the top
        textItem.setPos(static_x_position, static_y_position)

        # Update the linked Axis on range changes
        plotItem.sigRangeChanged.connect(lambda: linkedAxis.linkToView(plotItem.getViewBox()))

        # Inverse the y-axis
        plotItem.invertY(True)

        # Determine the scale system of logs and set the axis range accordingly
        if trackparams["scale"] == "log scale":

            plotItem.setLogMode(x=True, y=False)
            # Convert min_val and max_val to logarithmic scale if they are not zero
            if params['min_val'] > 0 and params['max_val'] > 0:
                log_min_val = math.log10(params['min_val'])
                log_max_val = math.log10(params['max_val'])
                plotItem.setXRange(log_min_val, log_max_val, padding=0)
        else:
            plotItem.setLogMode(x=False, y=False)  # Normal scale
            # Set the x-axis range as per the normal scale values
            plotItem.setXRange(params['min_val'], params['max_val'], padding=0)

        # Use invertX to reverse the direction of the X-axis
        if params['reverse']:
            plotItem.invertX(True)
            # Set fillLevel based on the 'filling_side' parameter
            if params["filling_side"] == "Left":
                fillLevel = params['max_val']
            else:
                fillLevel = params['min_val']
        else:
            plotItem.invertX(False)
            # Set fillLevel based on the 'filling_side' parameter
            if params["filling_side"] == "Right":
                fillLevel = params['max_val']
            else:
                fillLevel = params['min_val']

        color = pg.mkColor(params["line_color"])
        color.setAlpha(params["line_alpha"])

        color_map = pg.colormap.getFromMatplotlib(params["fill_color_palette"])
        color_map.reverse()
        brush = color_map.getBrush(span=(curve_min, params['max_val']), orientation='horizontal')
        gradient_pen = color_map.getPen(span=(curve_min, params['max_val']), width=trackparams[
            "major_line_width"] if "major_line_width" in trackparams else 5, orientation='horizontal')
        style = self.pen_styles.get(params["line_style"], Qt.PenStyle.SolidLine)
        if trackName == "FACIES":
            # Create the color palette
            colormap = matplotlib.colormaps["tab20"]

            # Generate the color palette
            color_palette = [QColor(*[int(255 * ch) for ch in colormap(i / colormap.N)[:3]]) for i in range(colormap.N)]

            # Create brushes for each bar
            brushes = [pg.mkBrush(color_palette[int(facies_type) - 1 % len(color_palette)]) for facies_type in
                       curve_data]

            # Constants for bar height and initial x-position
            bar_height = 0.5  # Match the interval between depth measurements
            x_start = 0  # Starting x-position of the bars

            # Prepare the parameters for BarGraphItem
            x0 = np.full(len(curve_data), x_start)
            y0 = depth_data - 0.25  # Center each bar in its depth interval
            widths = np.array(curve_data)  # Widths based on facies_type
            heights = np.full(len(curve_data), bar_height)
            pen_color = pg.mkColor(QColor(0, 0, 0, 0))
            pen_color.setAlpha(0)
            # Create and add the horizontal bar item
            bars = pg.BarGraphItem(x0=x0, y0=y0, height=heights, width=widths, brushes=brushes,
                                   pen=pg.mkPen(color=pen_color, width=0))
            plotItem.addItem(bars)

        else:
            # Create a PlotDataItem for the curve and add it to the PlotItem
            logItem = pg.PlotDataItem(curve_data, depth_data, brush=brush if params["filling_side"] != "None" else None,
                                      pen=gradient_pen if params["filling_side"] != "None" else pg.mkPen(color=color,
                                                                                                         style=style,
                                                                                                         width=
                                                                                                         trackparams[
                                                                                                             "major_line_width"] if "major_line_width" in trackparams else 2),
                                      fillLevel=fillLevel, connect="finite")
            plotItem.addItem(logItem)
            # Store the logItem in the track_plot_items dictionary
            if 'logs' not in self.track_plot_items[trackName]:
                self.track_plot_items[trackName]['logs'] = {}
            self.track_plot_items[trackName]['logs'][col] = logItem

            # Handle fill, palette, cut-offs etc.
            # Handle cut-off fill (filling method is still in development, using line method instead)
            # self._add_cutoff_fill(logItem, depth_data, curve_data, params)

            # Insert InfiniteLine objects to represent cut-offs
            self._add_cutoff_line(plotItem, trackName, trackparams["scale"], col, params)

        plotItem.update()

    def get_column_data(self, col_name):
        col_index = -1
        for i in range(self.table.columnCount()):
            if self.table.horizontalHeaderItem(i).text() == col_name:
                col_index = i
                break
        if col_index == -1:
            return []

        return [self.table.item(row, col_index).text() for row in range(self.table.rowCount())]

    def _add_cutoff_line(self, plotItem, trackName, scale, col, params):
        if 'left_cut_off' in params:
            if scale == "log scale":
                if params["left_cut_off"] == 0:
                    params["left_cut_off"] = 0.0001
                left_cut_off = np.log10(params["left_cut_off"])
            else:
                left_cut_off = params["left_cut_off"]
            left_cut_off_color = pg.mkColor(params["left_cut_off_color"])
            left_cut_off_color.setAlpha(params['left_cut_off_alpha'])
            left_cut_off_line = pg.InfiniteLine(pos=left_cut_off, angle=90,
                                                pen=pg.mkPen(color=left_cut_off_color, width=3))
            plotItem.addItem(left_cut_off_line)
            if 'left_cut_off' not in self.track_plot_items[trackName]:
                self.track_plot_items[trackName]['left_cut_off'] = {}
            self.track_plot_items[trackName]['left_cut_off'][col] = left_cut_off_line

        if 'right_cut_off' in params:
            if scale == "log scale":
                if params["right_cut_off"] == 0:
                    params["right_cut_off"] = 0.0001
                right_cut_off = np.log10(params["right_cut_off"])
            else:
                right_cut_off = params["right_cut_off"]
            right_cut_off_color = pg.mkColor(params["right_cut_off_color"])
            right_cut_off_color.setAlpha(params['right_cut_off_alpha'])
            right_cut_off_line = pg.InfiniteLine(pos=right_cut_off, angle=90,
                                                 pen=pg.mkPen(color=right_cut_off_color, width=3))
            plotItem.addItem(right_cut_off_line)
            if 'right_cut_off' not in self.track_plot_items[trackName]:
                self.track_plot_items[trackName]['right_cut_off'] = {}
            self.track_plot_items[trackName]['right_cut_off'][col] = right_cut_off_line

    def _add_cutoff_fill(self, logItem, depth_data, curve_data, params):
        curve_data_np = np.array(curve_data)
        depth_data_np = np.array(depth_data)

        # Function to find indices where the curve crosses the cutoff value
        def find_crossings(curve_data_np, depth_data_np, cutoff_val):
            crossings = []
            for i in range(1, len(curve_data_np)):
                if (curve_data_np[i - 1] < cutoff_val and curve_data_np[i] >= cutoff_val) or \
                        (curve_data_np[i - 1] >= cutoff_val and curve_data_np[i] < cutoff_val):
                    # Linear interpolation to find the depth of the crossing point
                    t = (cutoff_val - curve_data_np[i - 1]) / (curve_data_np[i] - curve_data_np[i - 1])
                    depth_at_crossing = depth_data_np[i - 1] + t * (depth_data_np[i] - depth_data_np[i - 1])
                    crossings.append(depth_at_crossing)
            return crossings

        # Function to create a polygon fill
        def create_polygon_fill(crossings, cutoff_val, color, alpha, left=True):
            if not crossings:
                return  # No crossings, no fill needed

            path = QPainterPath()

            # Start the path at the first crossing
            path.moveTo(cutoff_val, crossings[0])

            for i in range(len(crossings)):
                x_pos = curve_data_np[i] if left else cutoff_val
                path.lineTo(x_pos, depth_data_np[i])
                if i < len(crossings) - 1:
                    # Go to the next crossing at the cutoff value
                    path.lineTo(cutoff_val, crossings[i + 1])

            # Close the path at the last crossing
            path.lineTo(cutoff_val, crossings[-1])

            # Add the path as a QGraphicsPathItem to the plot
            color = pg.mkColor(color)
            color.setAlpha(int(alpha / 255.0))
            fillItem = QGraphicsPathItem(path)
            fillItem.setBrush(pg.mkBrush(color))
            logItem.getViewBox().addItem(fillItem)

        # Left cutoff fill
        if 'left_cut_off' in params:
            left_cutoff_val = params['left_cut_off']
            left_crossings = find_crossings(curve_data_np, depth_data_np, left_cutoff_val)
            create_polygon_fill(left_crossings, left_cutoff_val, params['left_cut_off_color'],
                                params['left_cut_off_alpha'], left=True)

        # Right cutoff fill
        if 'right_cut_off' in params:
            right_cutoff_val = params['right_cut_off']
            right_crossings = find_crossings(curve_data_np, depth_data_np, right_cutoff_val)
            create_polygon_fill(right_crossings, right_cutoff_val, params['right_cut_off_color'],
                                params['right_cut_off_alpha'], left=False)

    def _display_depth_track(self, col):
        # Find all columns except the depth column
        columns_except_depth = [c for c in self.df.columns if c != col]

        # Remove rows where any column has NaN values
        cleaned_df = self.df.dropna(subset=columns_except_depth, how='all')

        # Get the indices of the first and last non-NaN rows
        first_non_nan_index = cleaned_df.index[0]
        last_non_nan_index = cleaned_df.index[-1]

        # Retrieve only the depth data within this range
        depth_data = self.get_column_data(col)[first_non_nan_index:last_non_nan_index + 1]
        depth_data = list(map(float, depth_data))

        # Retrieve the visualization parameters for the DEPTH track
        params = self.all_parameters["DEPTH"]

        # Get the plotItem for the DEPTH track
        plotItem = self.track_plot_items["DEPTH"]['plot']
        # Clear existing TextItems
        for item in self.depthTextItems:
            plotItem.removeItem(item)
        self.depthTextItems.clear()

        # Inverse the y-axis
        plotItem.invertY(True)

        # Create the font object with desired properties
        Depth_font = QFont()
        Depth_font.setPointSizeF(self.all_parameters["DEPTH"].get("major_line_width", 12))
        Depth_font.setBold(True)  # Make it bold if required

        # Extracting parameters
        number_spacing = params["number_spacing"]
        tick_spacing = params["tick_spacing"]
        tick_spacing_int = int(math.ceil(tick_spacing))

        # Set tick spacing for both the left and right axes
        left_axis = plotItem.getAxis('left')
        right_axis = plotItem.getAxis('right')

        # Calculate ticks; start from the nearest lower multiple of tick_spacing_int
        min_depth = np.nanmin(depth_data)
        max_depth = np.nanmax(depth_data)

        # Find the nearest lower multiple of tick_spacing_int to min_depth
        start_tick = min_depth - (min_depth % tick_spacing_int)
        if start_tick > min_depth:
            start_tick -= tick_spacing_int

        # Find the nearest upper multiple of tick_spacing_int to max_depth
        end_tick = max_depth + (tick_spacing_int - (max_depth % tick_spacing_int))
        if end_tick < max_depth:
            end_tick += tick_spacing_int

        # Generate the list of ticks
        ticks = list(range(int(start_tick), int(end_tick) + 1, tick_spacing_int))

        major_ticks = [(depth, str(depth)) for depth in ticks]

        left_axis.setTicks([major_ticks, []])  # Second list is for minor ticks, which we're ignoring here
        right_axis.setTicks([major_ticks, []])

        left_axis.setStyle(tickLength=-10)
        right_axis.setStyle(tickLength=-10)

        params = self.all_parameters["DEPTH"]["curves"][col]
        color = pg.mkColor(params["line_color"])
        color.setAlpha = params["line_alpha"]

        # Display depth values; this is essentially labeling every nth depth value.
        for depth in depth_data:
            if depth % number_spacing == 0:  # Display depth every defined interval
                text_item = pg.TextItem(text="{:d}".format(int(depth)), color=color, anchor=(0.5, 0.5), angle=0,
                                        border=None, fill=None)
                text_item.setFont(Depth_font)  # Set the font to the text item
                plotItem.addItem(text_item)
                # Set the position of the text item based on depth
                text_item.setPos(0.5, depth)
                self.depthTextItems.append(text_item)
                if 'logs' not in self.track_plot_items["DEPTH"]:
                    self.track_plot_items["DEPTH"]['logs'] = {}

                if col not in self.track_plot_items["DEPTH"]['logs']:
                    self.track_plot_items["DEPTH"]['logs'][col] = []

                self.track_plot_items['DEPTH']['logs'][col].append(text_item)

        # plotItem.setYRange(min_depth, max_depth)

        # Disable the mouse interaction for x-axis to prevent horizontal scaling
        plotItem.setMouseEnabled(x=False, y=True)

        plotItem.update()


class AnchoredTextItem(pg.TextItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def viewRangeChanged(self):
        # Override the method to prevent auto-centering
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = LASApp()
    sys.exit(app.exec())
