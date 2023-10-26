import PySimpleGUI as sg


def create_main_gui():
    # Well Scraper section
    well_scraper_layout = [
        [sg.Text('Well Scraper')],
        [sg.Text('Placeholder for future content')]
    ]

    # File Management section
    file_management_layout = [
        [sg.Text('File Management')],
        [sg.Button('Open File Copying')],
        [sg.Button('Open Well Filter')]
    ]

    # Log Reader section
    log_reader_layout = [
        [sg.Text('Log Reader')],
        [sg.Button('Open DLIS Viewer')],
        [sg.Button('Open LAS Viewer')]
    ]

    # Sub-tabs for Data Management
    dm_tab_layout = [
        [sg.TabGroup([
            [sg.Tab('Well Scraper', well_scraper_layout, key='-WellScraper-')],
            [sg.Tab('File Management', file_management_layout, key='-FileMgmt-')],
            [sg.Tab('Log Reader', log_reader_layout, key='-LogReader-')]
        ],  tab_location='left')]
    ]

    # Main layout with main tab
    layout = [
        [sg.TabGroup([
            [sg.Tab('Data Management', dm_tab_layout, key='-DM-')]
            # Add more main tabs here...
        ])]
    ]

    # Create window
    window = sg.Window('MoPanda - Modular Petrophysics and Data Analysis Tool', layout, resizable=True)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        elif event == 'Open File Copying':
            # open_copy_tab()
            pass
        elif event == 'Open Well Filter':
            # wellfilter.open_well_filter_window()
            pass
        elif event == 'Open DLIS Viewer':
            # open_dlis_viewer_tab()
            pass
        elif event == 'Open LAS Viewer':
            # open_las_viewer_tab()
            pass

    window.close()


if __name__ == "__main__":
    create_main_gui()
