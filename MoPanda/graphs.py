# -*- coding: utf-8 -*-
"""
Graphs is a simple log viewer using matplotlib to create tracks of log
data. Allows graphically editing curve data through manual changes and
bulk shifting.

"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from matplotlib.backend_tools import ToolBase, ToolToggleBase
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.rcParams['backend'] = 'TkAgg'
plt.rcParams['toolbar'] = 'toolmanager'

class LogViewer(object):
    """LogViewer

    Uses matplotlib to create a figure and axes to display log data.
    XML templates are required to display curve data, with a few
    defaults provided.

    """

    def __init__(self, log, template_xml_path=None,
                 template_defaults=None, top=None, height=None, masking=None):
        self.log = log
        self.template_xml_path = template_xml_path
        self.template_defaults = template_defaults
        self.top = top
        self.height = height
        self.masking = masking

        ### private parameters for graphically editing curves ###

        # stores display names of curves
        self._edit_curve = None

        # stores matplotlib line objects by display name
        self._edit_curve_lines = {}

        # stores bool to show down-click to edit curves
        self._edit_lock = False

        # display name to log column name dictionary
        self._display_name_to_curve_name = {}

        default_templates_paths = {
            'raw': 'default_raw_template.xml',
            'full': 'default_full_template.xml',
            'electrofacies': 'default_electrofacies_template.xml',
            'salinity': 'default_salinity_template.xml',
            'permeability': 'default_permeability_template.xml'
        }

        file_dir = os.path.dirname(__file__)
        if template_xml_path is None and template_defaults is None:
            template_xml_path = os.path.join(file_dir, 'data/template',
                                             'default_raw_template.xml')

        elif template_xml_path is None and \
                template_defaults is not None:
            if template_defaults in default_templates_paths:
                file_name = default_templates_paths[template_defaults]
            else:
                print('template_defaults paramter must be in:')
                for key in default_templates_paths:
                    print(key)
                raise ValueError("%s is not valid template_defaults \
                                 parameter" % template_defaults)
            template_xml_path = os.path.join(file_dir, 'data/template',
                                             file_name)

        elif template_xml_path is not None and \
                template_defaults is None:
            template_xml_path = template_xml_path
        else:
            raise ValueError("template_xml_path and template_defaults \
                              cannot be specified at the same time.")

        with open(template_xml_path, 'r') as f:
            root = ET.fromstring(f.read())

        if 'top' in root.attrib and top is None:
            top = float(root.attrib['top'])
        elif top is None:
            top = self.log[0].min()

        if 'height' in root.attrib and height is None:
            bottom = top + float(root.attrib['height'])
        elif height is None:
            bottom = top + 500
        else:
            bottom = top + height

        ### format matplotlib figure ###
        tracks = root.findall('track')
        formations = root.findall('formation')
        num_tracks = len(tracks)

        print(formations)
        self.fig, self.axes = plt.subplots(1, num_tracks + 2,
                                           sharey=True,
                                           subplot_kw={'projection': 'FreezeAxes'})

        ### add formation background color ###
        for formation in formations:

            if 'name' in formation.attrib:
                name = formation.attrib['name']
            else:
                raise ValueError('Formation name required in \
                                  template %s' % template_xml_path)

            if 'color' in formation.attrib:
                color = formation.attrib['color']
            else:
                raise ValueError('Color required for formation %s \
                           in template %s' % (name, template_xml_path))

            if 'alpha' in formation.attrib:
                alpha = float(formation.attrib['alpha'])
            else:
                alpha = 0.1

            formation_top = self.log.tops[name]
            formation_bottom = self.log.formation_bottom_depth(name)
            formation_mid = (formation_top + formation_bottom) / 2.0
            for ax in self.axes:
                ax.axhspan(formation_top, formation_bottom,
                           facecolor=color, alpha=alpha)

            for ax in (self.axes[0], self.axes[-1]):
                ax.text(0.5, formation_mid, name,
                        verticalalignment='center',
                        horizontalalignment='center',
                        rotation=0, fontsize=9)

        numbers = None
        ticks = None
        track_widths = [0.5]

        depth_track_numbers = []
        track_names = []
        max_track_curves = 0
        c = 0
        for t, track in enumerate(tracks):

            ax = self.axes[t + 1]

            scale = 'linear'

            if 'display_name' in track.attrib:
                track_display_name = track.attrib['display_name']
                track_names.append(track_display_name)
            else:
                track_names.append(None)
                track_display_name = ''

            if 'scale' in track.attrib:
                scale = track.attrib['scale']
                ax.set_xscale(scale, nonpositive='clip')

            track_width = 1
            if 'width' in track.attrib:
                track_width = float(track.attrib['width'])
            track_widths.append(track_width)

            if track_display_name == 'DEPTH':

                ax.set_xlim(0, 1)
                depth_track_numbers.append(t + 1)

                if 'tick_spacing' in track.attrib:
                    tick_spacing = int(track.attrib['tick_spacing'])
                else:
                    raise ValueError('tick_spacing is required for \
                                                         depth track.')

                if 'number_spacing' in track.attrib:
                    number_spacing = int(track.attrib['number_spacing'])
                else:
                    raise ValueError('number_spacing is required for \
                                                         depth track.')

                if 'line_spacing' in track.attrib:
                    line_spacing = int(track.attrib['line_spacing'])
                else:
                    raise ValueError('line_spacing is required for \
                                                         depth track.')

                font_size = 16
                if 'font_size' in track.attrib:
                    font_size = float(track.attrib['font_size'])

                max_depth = self.log[0].max()

                ticks = range(0, int(max_depth) + tick_spacing,
                              tick_spacing)

                numbers = range(0, int(max_depth) + number_spacing,
                                number_spacing)

                lines = range(0, int(max_depth) + line_spacing,
                              line_spacing)

                for n in numbers:
                    ax.text(0.5, n, str(int(n)),
                            horizontalalignment='center',
                            verticalalignment='center',
                            clip_on=True,
                            fontsize=font_size)

            elif 'cumulative' in track.attrib:

                if 'left' not in track.attrib or \
                        'right' not in track.attrib:
                    raise ValueError('left and right values must be \
                     specified in cumulative tracks for template %s.' \
                                     % template_xml_path)

                left = float(track.attrib['left'])
                right = float(track.attrib['right'])

                invert_axis = False
                if right < left:
                    left, right = right, left
                    invert_axis = True

                summation = np.asarray(self.log[0] * 0)
                for c, curve in enumerate(track):
                    ### names ###
                    if 'curve_name' not in curve.attrib:
                        raise ValueError('Curve Name required in \
                                   template at %s' % template_xml_path)
                    curve_name = curve.attrib['curve_name']

                    if curve_name not in self.log.keys():
                        raise ValueError('Curve %s not found in log.' \
                                         % curve_name)

                    if 'fill_color' not in curve.attrib:
                        raise ValueError('Curve fill_color must be \
                                         specificied for cumulative \
                                         track in template at %s' % \
                                         template_xml_path)

                    fill_color = curve.attrib['fill_color']

                    increase = summation + np.asarray(self.log[curve_name])

                    ax.fill_betweenx(self.log[0],
                                     summation,
                                     increase,
                                     color=fill_color)

                    summation = increase

                    ### display ###
                    if invert_axis:
                        x = (len(track) - float(c)) / len(track) - 1.0 / \
                            (2 * len(track))
                    else:
                        x = float(c) / len(track) + 1.0 / (2 * len(track))
                    display_name = curve.attrib.get('display_name', curve.attrib['curve_name'])
                    ax.text(x, 1.05, display_name, rotation=45,
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            transform=ax.transAxes,
                            fontsize=9,
                            color=fill_color)

                if 'major_lines' in track.attrib:
                    num_major_lines = int(track.attrib['major_lines']) + 1
                    dist = abs(left - right) / num_major_lines
                    major_lines = np.arange(left + dist, right, dist)
                    for m in major_lines:
                        ax.plot((m, m), (0, self.log[0].max()),
                                color='#c0c0c0', lw=0.5)

                if track.attrib['display_name'] == "WATER PARTITION":
                    ax.text(0, 0.98 + 0.035 * (len(curve) + 1),
                            left,
                            horizontalalignment='left',
                            verticalalignment='bottom',
                            transform=ax.transAxes,
                            fontsize=12,
                            color='#000000')

                    ax.text(1, 0.98 + 0.035 * (len(curve) + 1),
                            right,
                            horizontalalignment='right',
                            verticalalignment='bottom',
                            transform=ax.transAxes,
                            fontsize=12,
                            color='#000000')

                ax.set_xlim(left, right)
                if invert_axis:
                    ax.invert_xaxis()

                c = int(c / 2) + 1

            # template for 'distribution curve' display, for example, T1/ T2 distribution.

            elif 'distribution' in track.attrib:
                total_curves = len(track)  # Total number of curves in the track
                track_width = float(track.attrib['width'])  # Width of the track

                # Find the maximum value across all curves within the track, ignoring the top 5% of data
                values = []
                for curve in track:
                    curve_name = curve.attrib.get('curve_name')
                    if curve_name and curve_name in self.log.keys():
                        curve_data = self.log[curve_name]
                        curve_data = np.where(np.isnan(curve_data), 0, curve_data)
                        values.extend(curve_data)

                values = np.array(values)
                top_5_percentile = np.percentile(values, 90)  # Get the 95th percentile value
                values = values[values <= top_5_percentile]  # Filter values below the 95th percentile
                max_value = np.max(values)
                max_value = np.ceil(max_value * 100) / 100  # Round up the maximum value with 0.01 precision

                # Create an empty 2D matrix for the heatmap

                heatmap = np.zeros((len(self.log[0]), total_curves))

                for c, curve in enumerate(track):
                    curve_name = curve.attrib.get('curve_name')
                    if curve_name and curve_name in self.log.keys():
                        curve_data = self.log[curve_name]
                        curve_data = np.where(np.isnan(curve_data), 0, curve_data)
                        heatmap[:, c] = curve_data

                # Assign colors using the YlGn color palette
                cmap = plt.cm.YlGn
                norm = plt.Normalize(vmin=0, vmax=max_value)

                # Plot the heatmap
                im = ax.imshow(heatmap, aspect='auto', cmap=cmap, interpolation='none', vmin=0, vmax=max_value,
                               extent=[0, total_curves, self.log[0].max(), 0])

                # Set the colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.3, pad=0.001)
                # cbar.set_label('Values')

                # Hide the axis labels
                ax.set_xticks([])
                ax.set_yticks([])

            else:

                for c, curve in enumerate(track):
                    ### names ###
                    if 'curve_name' in curve.attrib:
                        if masking.get('status') and curve.attrib['curve_name'] in masking.get('curves_to_mask'):
                            original_name = curve.attrib['curve_name']
                            curve_name = f'{original_name}_masked'
                            print(curve_name)
                        else:
                            curve_name = curve.attrib['curve_name']
                    else:
                        raise ValueError('Curve Name required in \
                                   template at %s' % template_xml_path)

                    if curve_name not in self.log.keys():
                        raise ValueError('Curve %s not found in log.' \
                                         % curve_name)

                    ### style and scale ###

                    left = None
                    right = None
                    if 'left' in curve.attrib:
                        left = float(curve.attrib['left'])
                        left_label = ' ' + curve.attrib['left']
                    if 'right' in curve.attrib:
                        right = float(curve.attrib['right'])
                        right_label = curve.attrib['right'] + ' '

                    if left is None:
                        print('Adjust template at %s.' % \
                              template_xml_path)
                        raise ValueError('Left X axis not found for \
                                          curve %s.' % curve_name)
                    if right is None:
                        print('Adjust template at %s.' \
                              % template_xml_path)

                        raise ValueError('Right X axis not found for \
                                         curve %s.' % curve_name)

                    line_style = '-'
                    color = '#000000'
                    width = 1
                    alpha = 1
                    marker = None
                    marker_size = 0

                    if 'line_style' in curve.attrib:
                        line_style = curve.attrib['line_style']
                    if 'color' in curve.attrib:
                        color = curve.attrib['color']
                    if 'width' in curve.attrib:
                        width = float(curve.attrib['width'])
                    if 'alpha' in curve.attrib:
                        alpha = float(curve.attrib['alpha'])
                    if 'marker' in curve.attrib:
                        marker = curve.attrib['marker']
                    if 'marker_size' in curve.attrib:
                        marker_size = float(curve.attrib['marker_size'])

                    if scale == 'log':
                        ax.set_xlim(left, right)
                        x = self.log[curve_name]
                        m = None
                        b = None

                        if 'left_color_value' in curve.attrib:
                            left_color_value = \
                                float(curve.attrib['left_color_value'])
                        else:
                            left_color_value = left

                        if 'right_color_value' in curve.attrib:
                            if curve.attrib['right_color_value'] == "auto":
                                right_color_value = len(set(x))
                            else:
                                right_color_value = \
                                    float(curve.attrib['right_color_value'])
                        else:
                            right_color_value = right

                    else:
                        ax.set_xlim(0, 1)
                        m = (1 - 0) / (right - left)
                        b = -m * left
                        x = m * self.log[curve_name] + b
                        left = 0
                        right = 1

                        if 'left_color_value' in curve.attrib:
                            left_color_value = \
                                float(curve.attrib['left_color_value'])
                            left_color_value = m * left_color_value + b
                        else:
                            left_color_value = left

                        if 'right_color_value' in curve.attrib:
                            if curve.attrib['right_color_value'] == "auto":
                                right_color_value = len(np.unique(x[~np.isnan(x)]))
                            else:
                                right_color_value = float(curve.attrib['right_color_value'])
                            right_color_value = m * right_color_value + b
                        else:
                            right_color_value = right

                    ### label ###
                    if 'display_name' in curve.attrib:
                        self._display_name_to_curve_name[curve.attrib['display_name']] = curve_name

                        ax.text(0.5, 0.98 + 0.035 * (c + 1),
                                curve.attrib['display_name'],
                                horizontalalignment='center',
                                verticalalignment='bottom',
                                transform=ax.transAxes,
                                fontsize=12,
                                color=color,
                                picker=True)
                        if not track.attrib['display_name'] == "MINERALOGY" and not 'FACIES' in curve.attrib[
                            'curve_name']:
                            ax.text(0, 0.98 + 0.035 * (c + 1),
                                    left_label,
                                    horizontalalignment='left',
                                    verticalalignment='bottom',
                                    transform=ax.transAxes,
                                    fontsize=12,
                                    color=color)

                            ax.text(1, 0.98 + 0.035 * (c + 1),
                                    right_label,
                                    horizontalalignment='right',
                                    verticalalignment='bottom',
                                    transform=ax.transAxes,
                                    fontsize=12,
                                    color=color)

                    if 'fill' in curve.attrib:

                        fill_color = '#000000'
                        if 'fill_color' in curve.attrib:
                            fill_color = curve.attrib['fill_color']

                            if curve.attrib['fill'] == 'left':
                                baseline = left
                            elif curve.attrib['fill'] == 'right':
                                baseline = right

                            ax.fill_betweenx(self.log[0],
                                             baseline,
                                             x,
                                             color=fill_color)

                        elif 'fill_color_map' in curve.attrib:
                            cmap_name = curve.attrib['fill_color_map']
                            cmap = plt.get_cmap(cmap_name)
                            if len(np.unique(x)) < 50:
                                color_index = np.unique(x[~np.isnan(x)])
                                print(color_index)
                            else:
                                if curve.attrib['fill'] == 'left':
                                    color_index = np.linspace(left_color_value, right_color_value, num=50)
                                elif curve.attrib['fill'] == 'right':
                                    color_index = np.linspace(right_color_value, left_color_value, num=50)

                            if curve.attrib['fill'] == 'left':
                                baseline = np.ones(len(x)) * min(left_color_value, left)
                            elif curve.attrib['fill'] == 'right':
                                baseline = np.ones(len(x)) * max(right_color_value, right)

                            for ci in sorted(color_index):
                                ci_normalized = (ci - right_color_value) / (left_color_value - right_color_value)
                                if curve.attrib['fill'] == 'right' and left_color_value > right_color_value:
                                    ci_normalized = 1 - ci_normalized
                                color = cmap(ci_normalized)
                                ax.fill_betweenx(self.log[0], baseline, x, where=x >= ci, color=color)

                    if 'right_cutoff_fill' in curve.attrib:

                        fill_color = '#000000'
                        if 'right_cutoff_fill_color' in curve.attrib:
                            fill_color = \
                                curve.attrib['right_cutoff_fill_color']

                        cutoff_value = \
                            float(curve.attrib['right_cutoff_fill'])

                        if scale != 'log':
                            v = m * cutoff_value + b
                        else:
                            v = cutoff_value
                        ax.fill_betweenx(self.log[0],
                                         v,
                                         x,
                                         color=fill_color,
                                         where=v < x)
                        ax.plot(v * np.ones(len(self.log[0][v < x])),
                                self.log[0][v < x], c="#000000",
                                lw=0.5)

                    if 'left_cutoff_fill' in curve.attrib:

                        fill_color = '#000000'
                        if 'left_cutoff_fill_color' in curve.attrib:
                            fill_color = \
                                curve.attrib['left_cutoff_fill_color']

                        cutoff_value = \
                            float(curve.attrib['left_cutoff_fill'])

                        if scale != 'log':
                            v = m * cutoff_value + b
                        else:
                            v = cutoff_value
                        ax.fill_betweenx(self.log[0],
                                         v,
                                         x,
                                         color=fill_color,
                                         where=x < v)

                        ax.plot(v * np.ones(len(self.log[0][x < v])),
                                self.log[0][x < v], c="#000000",
                                lw=0.5)

                    if 'left_crossover' in curve.attrib:

                        fill_color = '#000000'
                        if 'left_crossover_fill_color' in curve.attrib:
                            fill_color = \
                                curve.attrib['left_crossover_fill_color']

                        left_curve = curve.attrib['left_crossover']
                        if left_curve not in self.log.keys():
                            raise ValueError('Curve %s not found in \
                                             log.' % left_curve)

                        if 'left_crossover_left' not in curve.attrib \
                                and 'left_crossover_right' not in curve.attrib:
                            raise ValueError('left and right crossover \
                                     values not found in template %s' \
                                             % template_xml_path)

                        left_crossover_left = \
                            float(curve.attrib['left_crossover_left'])

                        left_crossover_right = \
                            float(curve.attrib['left_crossover_right'])

                        if scale != 'log':
                            m = (1 - 0) / (left_crossover_right - \
                                           left_crossover_left)

                            b = -m * left_crossover_left
                            v = m * self.log[left_curve] + b
                        else:
                            v = self.log[left_curve]

                        ax.fill_betweenx(self.log[0],
                                         x,
                                         v,
                                         color=fill_color,
                                         where=v < x)

                    if 'right_crossover' in curve.attrib:

                        fill_color = '#000000'
                        if 'right_crossover_fill_color' in curve.attrib:
                            fill_color = \
                                curve.attrib['right_crossover_fill_color']

                        left_curve = curve.attrib['right_crossover']
                        if left_curve not in self.log.keys():
                            raise ValueError('Curve %s not found in \
                                                    log.' % left_curve)

                        if 'right_crossover_left' not in curve.attrib \
                                and 'right_crossover_right' not in curve.attrib:
                            raise ValueError('left and right crossover \
                                     values not found in template %s' \
                                             % template_xml_path)

                        left_crossover_left = \
                            float(curve.attrib['right_crossover_left'])

                        left_crossover_right = \
                            float(curve.attrib['right_crossover_right'])

                        if scale != 'log':
                            m = (1 - 0) / (left_crossover_right - \
                                           left_crossover_left)
                            b = -m * left_crossover_left
                            v = m * self.log[left_curve] + b
                        else:
                            v = self.log[left_curve]

                        ax.fill_betweenx(self.log[0],
                                         x,
                                         v,
                                         color=fill_color,
                                         where=v > x)

                    curve_line = ax.plot(x,
                                         self.log[0],
                                         c=color,
                                         lw=width,
                                         ls=line_style,
                                         marker=marker,
                                         ms=marker_size)[0]

                    self._edit_curve_lines[curve_name] = \
                        (curve_line, m, b)

                if scale == 'log':
                    ax.xaxis.grid(True, which='both', color='#e0e0e0')

                elif 'major_lines' in track.attrib:
                    num_major_lines = \
                        int(track.attrib['major_lines']) + 1

                    dist = abs(left - right) / num_major_lines
                    major_lines = np.arange(left + dist, right, dist)
                    for m in major_lines:
                        ax.plot((m, m), (0, self.log[0].max()),
                                color='#c0c0c0', lw=0.5)

        if max_track_curves < 5:
            max_track_curves = 5
        else:
            max_track_curves += 1

        ### adjust track widths ###
        track_widths.append(0.5)
        track_widths = np.asarray(track_widths)
        track_widths = track_widths / np.sum(track_widths)
        track_locations = [0]
        for t in range(1, len(track_widths)):
            track_locations.append(track_locations[t - 1] + \
                                   track_widths[t - 1])

        for a, ax in enumerate(self.axes):
            post = ax.get_position()
            new_post = (track_locations[a], 0.01, track_widths[a],
                        post.height)
            ax.set_position(new_post)
            if ticks is not None:
                ax.set_yticks(lines, minor=False)
                ax.set_yticks(ticks, minor=True)
                ax.tick_params(axis='y', direction='inout',
                               length=6, width=1,
                               colors='#000000', which='minor')
                if a not in depth_track_numbers:
                    ax.yaxis.grid(True, which='major')
            else:
                ax.set_yticks([])
                ax.set_xticks([])

            if a > 0 and a < len(self.axes) - 1:
                track_title = track_names[a - 1]
                if track_title is not None:
                    track_title += '\n' * max_track_curves
                    ax.set_title(track_title, fontweight='bold')

        if top is not None and bottom is not None:
            plt.ylim((top, bottom))

        plt.gca().invert_yaxis()
        self.fig.set_size_inches(11, 8.5)

    def show(self):
        """
        Calls matplotlib.pyplot.show() to display log viewer. It
        includes options in the toolbar to graphically edit
        curve data, and stores these changes within the LogViewer
        object. After editing is finished, access the updated data.

        """

        if len(str(self.log.well['UWI'].value)) > 0:
            log_window_title = 'UWI: ' + str(self.log.well['UWI'].value)
        elif len(self.log.well['API'].value) > 0:
            log_window_title = 'API: ' + str(self.log.well['API'].value)
        else:
            log_window_title = 'Log Viewer'
        self.fig.canvas.manager.window.title(log_window_title)

        # add edit tools to toolmanager
        tm = self.fig.canvas.manager.toolmanager
        tm.add_tool('Curve Edit', _CurveEditToggle)
        tm.add_tool('Bulk Shift', _BulkShiftToggle)

        # add edit tools to toolbar
        tb = self.fig.canvas.manager.toolbar
        tb.add_tool('Curve Edit', 'Edit')
        tb.add_tool('Bulk Shift', 'Edit')

        # remove non-useful tools
        self.fig.canvas.manager.toolmanager.remove_tool('forward')
        self.fig.canvas.manager.toolmanager.remove_tool('back')
        self.fig.canvas.manager.toolmanager.remove_tool('help')

        self.fig.canvas.mpl_connect('pick_event', self._curve_pick)
        self.fig.canvas.mpl_connect('button_press_event',
                                    self._edit_lock_toggle)
        self.fig.canvas.mpl_connect('button_release_event',
                                    self._edit_lock_toggle)
        self.fig.canvas.mpl_connect('motion_notify_event',
                                    self._draw_curve)
        plt.show()

    def _curve_pick(self, event):
        """
        Event handler for selecting a curve to edit. Results in
        self._edit_curve being set to matplotlib text object. Connected
        on line 852 with 'pick_event'.
        """

        if self._edit_curve is not None:
            self._edit_curve.set_bbox({'facecolor': 'white',
                                       'edgecolor': 'white',
                                       'alpha': 0})

        self._edit_curve = event.artist
        draw = self.fig.canvas.manager.toolmanager.get_tool('Curve Edit')
        bulk = self.fig.canvas.manager.toolmanager.get_tool('Bulk Shift')

        if draw.toggled or bulk.toggled:
            self._edit_curve.set_bbox({'facecolor': 'khaki',
                                       'edgecolor': 'khaki',
                                       'alpha': 1})

        self.fig.canvas.draw()

    def _edit_lock_toggle(self, event):
        """
        Event handler to check for correct axis associated with
        selected curve. Will allow _draw_curve to function if click is
        in proper axis based on the self._edit_curve property set with
        _curve_pick. Connected on lines 853 and 855 to
        :code:`button_press_event` and :code:`button_release_event`.
        """

        if self._edit_curve and hasattr(event, 'inaxes'):
            if event.inaxes:
                ax_num = np.where(self.axes == event.inaxes)[0]
                if len(ax_num) > 0:
                    curve_num = \
                        np.where(self.axes == self._edit_curve.axes)[0]
                    if ax_num == curve_num:
                        self._edit_lock = not self._edit_lock

    def _draw_curve(self, event):
        """
        Event handler for changing data in the figure and in the log
        object. Connected on line 857 with :code:`motion_notify_event`.
        """

        draw = self.fig.canvas.manager.toolmanager.get_tool('Curve Edit')
        bulk = self.fig.canvas.manager.toolmanager.get_tool('Bulk Shift')

        if draw.toggled and self._edit_lock:
            x, y = event.xdata, event.ydata

            curve_name = \
                self._display_name_to_curve_name[self._edit_curve.get_text()]

            cursor_depth_index = np.argmin(np.abs(self.log[0] - y))
            line, m, b = self._edit_curve_lines[curve_name]

            x_data = line.get_xdata()
            x_data[cursor_depth_index] = x
            line.set_xdata(x_data)

            if m is not None and b is not None:
                x = (x - b) / m

            self.log[curve_name][cursor_depth_index] = x

            self.fig.canvas.draw()

        elif bulk.toggled and self._edit_lock:
            x, y = event.xdata, event.ydata

            curve_name = \
                self._display_name_to_curve_name[self._edit_curve.get_text()]

            cursor_depth_index = np.argmin(np.abs(self.log[0] - y))

            line, m, b = self._edit_curve_lines[curve_name]

            x_data = line.get_xdata()
            x_diff = x_data[cursor_depth_index] - x
            x_data = x_data - x_diff
            line.set_xdata(x_data)

            if m is not None and b is not None:
                x_data = (x_data - b) / m

            self.log[curve_name][:] = x_data

            self.fig.canvas.draw()


class _CurveEditToggle(ToolToggleBase):
    """
    Curve edit toggle for toolbar. Allows redrawing curves graphically
    in matplotlib.
    """
    description = 'Curve Draw Edit'

    file_dir = os.path.dirname(__file__)
    image = os.path.join(file_dir, 'data', 'images', 'draw.png')

    def enable(self, event):
        """
        Color previously selected curve to khaki when editing
        """
        for obj in gc.get_objects():
            if isinstance(obj, LogViewer):
                if obj._edit_curve is not None:
                    obj._edit_curve.set_bbox({'facecolor': 'khaki',
                                              'edgecolor': 'khaki',
                                              'alpha': 1})
                    obj.fig.canvas.draw()

    def disable(self, event):
        """
        Color previously selected curve to white when not editing
        """
        for obj in gc.get_objects():
            if isinstance(obj, LogViewer):
                if obj._edit_curve is not None:
                    obj._edit_curve.set_bbox({'facecolor': 'white',
                                              'edgecolor': 'white',
                                              'alpha': 0})
                    obj.fig.canvas.draw()


class _BulkShiftToggle(ToolToggleBase):
    """
    Bulk shift toggle for toolbar. Allows bulk shifting curves
    graphically in matplotlib.
    """
    description = 'Bulk Shift Edit'

    file_dir = os.path.dirname(__file__)
    image = os.path.join(file_dir, 'data', 'images', 'bulk.png')

    def enable(self, event):
        """
        Color previously selected curve to khaki when editing
        """
        for obj in gc.get_objects():
            if isinstance(obj, LogViewer):
                if obj._edit_curve is not None:
                    obj._edit_curve.set_bbox({'facecolor': 'khaki',
                                              'edgecolor': 'khaki',
                                              'alpha': 1})
                    obj.fig.canvas.draw()

    def disable(self, event):
        """
        Color previously selected curve to white when not editing
        """
        for obj in gc.get_objects():
            if isinstance(obj, LogViewer):
                if obj._edit_curve is not None:
                    obj._edit_curve.set_bbox({'facecolor': 'white',
                                              'edgecolor': 'white',
                                              'alpha': 0})
                    obj.fig.canvas.draw()


class _FreezeAxes(mpl.axes.Axes):
    """
    FreezeAxes is a custom axes class that disallows scrolling on the x-axis while
    allowing scrolling along the y-axis. This method leverages the event handling mechanism in matplotlib to
    intercept and modify scrolling events.
    """
    name = 'FreezeAxes'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._disable_x_scroll()

    def _disable_x_scroll(self):
        self.set_navigate_mode("pan")
        self._pan_kx = 0.0

    def scroll_pan(self, event):
        if event.key == "x":
            event.key = None
        super().scroll_pan(event)


mpl.projections.register_projection(_FreezeAxes)
