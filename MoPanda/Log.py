import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import datetime as dt
from scipy.optimize import nnls

from lasio import LASFile, CurveItem


def check_file(output_file):
    if os.path.exists(output_file):
        while True:
            response = input(
                f"The output file '{output_file}' already exists. Do you want to overwrite it? (Y/N): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                return False
            else:
                print("Invalid response. Please enter 'Y' or 'N'.")
    return True


# Major Log subclass that inherits the LASFile superclass and perform expansive mathematical petrophysics analysis and
# data analysis.
class Log(LASFile):
    def __init__(self, file_ref=None, **kwargs):
        if file_ref is not None:
            super().__init__(file_ref=file_ref, null_policy='strict',
                             autodetect_encoding=True, **kwargs)
        self.filename = os.path.splitext(os.path.basename(file_ref))[0]
        # self.aliasing()  # Uncomment this line if auto-aliasing during the loading stage of data is wanted
        self.fluid_properties_params = None
        self.multimineral_parameters = None
        self.tops = {}

    def log_qc(self, start_depth, end_depth):
        preserved_columns = []
        df = self.df().reset_index()
        n_total_logs = len(df.columns)
        for column in df.columns:
            column_data = df[column]
            column_start_depth = column_data.first_valid_index()
            column_end_depth = column_data.last_valid_index()

            if column_start_depth <= start_depth and column_end_depth >= end_depth:
                preserved_columns.append(column)

        preserved_df = df[preserved_columns]
        n_preserved_logs = len(preserved_df.columns)
        print("Total log number:", n_total_logs)
        print("Selected", n_preserved_logs, "logs")
        print("Dropped", n_total_logs - n_preserved_logs, "logs")

        # Delete dropped curves from LASFile
        for curve_name in self.keys():
            if curve_name not in preserved_columns:
                self.delete_curve(curve_name)
        return preserved_df

    def aliasing(self):
        """
        Curve Alias is provided by the log_alias.xml file
        """

        file_dir = os.path.dirname(__file__)
        alias_path = os.path.join(file_dir, 'data', 'log_alias.xml')

        if not os.path.isfile(alias_path):
            raise ValueError('No alias file at: %s' % alias_path)

        with open(alias_path, 'r') as f:
            root = ET.fromstring(f.read())

        for alias in root:
            for curve in alias:
                original_curve = curve.tag
                new_curve = alias.tag

                if original_curve in self.keys() and new_curve not in self.keys():
                    curve_item = self.curves[original_curve]
                    self.append_curve(new_curve, self[original_curve],
                                      unit=curve_item.unit, value=curve_item.value,
                                      descr=curve_item.descr)

    def load_tops(self, csv_path=None, depth_type='MD', source=None):
        if csv_path is None:
            local_path = os.path.dirname(__file__)
            csv_path = os.path.join(local_path, 'data', 'tops.csv')

        top_df = pd.read_csv(csv_path, dtype={'UWI': str, 'Src': str})

        # Change data type to str for columns starting with 'Fm Name'
        fm_columns = [col for col in top_df.columns if col.startswith('Fm Name')]
        form = fm_columns[0]

        if source and source != 'All':
            # Check if the source exists in the 'Src' column, if not exit
            if source not in top_df['Src'].unique():
                print("Source not found in the 'Src' column.")
                return
            top_df = top_df[top_df['Src'] == source]

        columns_to_select = ['UWI', form, depth_type]
        top_df = top_df[columns_to_select]
        top_df = top_df.dropna(subset=[depth_type])

        # Process formation names based on the source value
        top_df[form] = top_df[form].str.rstrip(' *')

        # If source is 'CA', then remove the CA_ Prefix.
        # Can be modified or added based on demands.
        if source == 'CA':
            top_df[form] = top_df[form].str.lstrip('CA_')

        # Only select tops for specific well/UWI.
        well_tops_df = top_df[top_df['UWI'] == str(self.well['UWI'].value)]
        for r, row in well_tops_df.iterrows():
            self.tops[row[form]] = float(row[depth_type].replace(",", ""))

    def formation_bottom_depth(self, formation):
        """
        Return the bottom depth of specified formation/top of formation below specified formation.
        """

        top = self.tops[formation]

        bottom = np.max(self[0])
        formation_below = bottom - top
        for form in self.tops:
            form_top_depth = self.tops[form]
            if form_top_depth > top and form_top_depth - top < formation_below:
                bottom = form_top_depth
                formation_below = bottom - top

        return bottom

    def load_fluid_properties(self, csv_path=None):
        """
        Reads parameters from a csv for input into fluid properties.

        This method reads the file located at the csv_path and turns the
        values into dictionaries to be used as inputs into the
        fluid_properties method.

        This reference_ is a sample csv file with default data for
        fluid properties.

        """

        if csv_path is None:
            local_path = os.path.dirname(__file__)
            csv_path = os.path.join(local_path, 'data', 'fluid_properties.csv')

        df = pd.read_csv(csv_path)
        df = df.set_index('name')

        self.fluid_properties_params = df.to_dict(orient='index')

    def fluid_properties(self, top=0, bottom=100000, mast=67,
                         temp_grad=0.015, press_grad=0.5, rws=0.1, rwt=70,
                         rmfs=0.4, rmft=100, gas_grav=0.67, oil_api=38, p_sep=100,
                         t_sep=100, yn2=0, yco2=0, yh2s=0, yh20=0, rs=0,
                         lith_grad=1.03, biot=0.8, pr=0.25):
        # fluid property calculations
        depth_index = np.intersect1d(np.where(self[0] >= top)[0],
                                     np.where(self[0] < bottom)[0])
        depths = self[0][depth_index]

        form_temp = mast + temp_grad * depths
        pore_press = press_grad * depths

        # water properties
        rw = (rwt + 6.77) / (form_temp + 6.77) * rws
        rmf = (rmft + 6.77) / (form_temp + 6.77) * rmfs

        rw68 = (rwt + 6.77) / (68 + 6.77) * rws
        rmf68 = (rmft + 6.77) / (68 + 6.77) * rws

        # weight percent total dissolved solids
        xsaltw = 10 ** (-0.5268 * (np.log10(rw68)) ** 3 - 1.0199 * (np.log10(rw68)) ** 2 - 1.6693 * (
            np.log10(rw68)) - 0.3087)
        xsaltmf = 10 ** (-0.5268 * (np.log10(rmf68)) ** 3 - 1.0199 * (np.log10(rmf68)) ** 2 - 1.6693 * (
            np.log10(rmf68)) - 0.3087)

        # bw for reservoir water
        dvwt = -1.0001 * 10 ** -2 + 1.33391 * 10 ** -4 * form_temp + \
               5.50654 * 10 ** -7 * form_temp ** 2

        dvwp = -1.95301 * 10 ** -9 * pore_press * form_temp - \
               1.72834 * 10 ** -13 * pore_press ** 2 * form_temp - \
               3.58922 * 10 ** -7 * pore_press - \
               2.25341 * 10 ** -10 * pore_press ** 2

        bw = (1 + dvwt) * (1 + dvwp)

        # calculate solution gas in water ratio
        rsa = 8.15839 - 6.12265 * 10 ** -2 * form_temp + \
              1.91663 * 10 ** -4 * form_temp ** 2 - \
              2.1654 * 10 ** -7 * form_temp ** 3

        rsb = 1.01021 * 10 ** -2 - 7.44241 * 10 ** -5 * form_temp + \
              3.05553 * 10 ** -7 * form_temp ** 2 - \
              2.94883 * 10 ** -10 * form_temp ** 3

        rsc = -1.0 * 10 ** -7 * (
                9.02505 - 0.130237 * form_temp + 8.53425 * 10 ** -4 * form_temp ** 2 - 2.34122 * 10 ** -6 * form_temp ** 3 + 2.37049 * 10 ** -9 * form_temp ** 4)

        rswp = rsa + rsb * pore_press + rsc * pore_press ** 2
        rsw = rswp * 10 ** (-0.0840655 * xsaltw * form_temp ** -0.285584)

        # log responses
        rho_w = (2.7512 * 10 ** -5 * xsaltw + 6.9159 * 10 ** -3 * xsaltw + 1.0005) * bw

        rho_mf = (2.7512 * 10 ** -5 * xsaltmf + 6.9159 * 10 ** -3 * xsaltmf + 1.0005) * bw

        nphi_w = 1 + 0.4 * (xsaltw / 100)
        nphi_mf = 1 + 0.4 * (xsaltmf / 100)

        # net effective stress
        nes = ((lith_grad * depths) - (biot * press_grad * depths) + 2 * (pr / (1 - pr)) * (lith_grad * depths) - (
                biot * press_grad * depths)) / 3

        # gas reservoir
        if oil_api == 0:
            # hydrocarbon gravity only
            hc_grav = (gas_grav - 1.1767 * yh2s - 1.5196 * yco2 - 0.9672 * yn2 - 0.622 * yh20) / \
                      (1.0 - yn2 - yco2 - yh20 - yh2s)

            # pseudo-critical properties of hydrocarbon
            ppc_h = 756.8 - 131.0 * hc_grav - 3.6 * (hc_grav ** 2)
            tpc_h = 169.2 + 349.5 * hc_grav - 74.0 * (hc_grav ** 2)

            # pseudo-critical properties of mixture
            ppc = (1.0 - yh2s - yco2 - yn2 - yh20) * ppc_h + \
                  1306.0 * yh2s + 1071.0 * yco2 + \
                  493.1 * yn2 + 3200.1 * yh20

            tpc = (1.0 - yh2s - yco2 - yn2 - yh20) * tpc_h + \
                  672.35 * yh2s + 547.58 * yco2 + \
                  227.16 * yn2 + 1164.9 * yh20

            # Wichert-Aziz correction for H2S and CO2
            if yco2 > 0 or yh2s > 0:
                epsilon = 120 * ((yco2 + yh2s) ** 0.9 - (yco2 + yh2s) ** 1.6) + \
                          15 * (yh2s ** 0.5 - yh2s ** 4)

                tpc_temp = tpc - epsilon
                ppc = (ppc_a * tpc_temp) / \
                      (tpc + (yh2s * (1.0 - yh2s) * epsilon))

                tpc = tpc_temp
            # Casey's correction for nitrogen and water vapor
            if yn2 > 0 or yh20 > 0:
                tpc_cor = -246.1 * yn2 + 400 * yh20
                ppc_cor = -162.0 * yn2 + 1270.0 * yh20
                tpc = (tpc - 227.2 * yn2 - 1165.0 * yh20) / \
                      (1.0 - yn2 - yh20) + tpc_cor

                ppc = (ppc - 493.1 * yn2 - 3200.0 * yh20) / \
                      (1.0 - yn2 - yh20) + ppc_cor

            # Reduced pseudo-critical properties
            tpr = (form_temp + 459.67) / tpc
            ppr = pore_press / ppc

            # z factor from Dranchuk and Abou-Kassem fit of standing and Katz chart
            a = [0.3265,
                 -1.07,
                 -0.5339,
                 0.01569,
                 -0.05165,
                 0.5475,
                 -0.7361,
                 0.1844,
                 0.1056,
                 0.6134,
                 0.721]

            t2 = a[0] * tpr + a[1] + a[2] / (tpr ** 2) + \
                 a[3] / (tpr ** 3) + a[4] / (tpr ** 4)

            t3 = a[5] * tpr + a[6] + a[7] / tpr
            t4 = -a[8] * (a[6] + a[7] / tpr)
            t5 = a[9] / (tpr ** 2)

            r = 0.27 * ppr / tpr
            z = 0.27 * ppr / tpr / r

            counter = 0
            diff = 1
            while counter <= 10 and diff > 10 ** -5:
                counter += 1

                f = r * (tpr + t2 * r + t3 * r ** 2 + t4 * r ** 5 + t5 * r ** 2 * (1 + a[10] * r ** 2) * np.exp(
                    -a[10] * r ** 2)) - 0.27 * ppr

                fp = tpr + 2 * t2 * r + 3 * t3 * r ** 2 + \
                     6 * t4 * r ** 5 + t5 * r ** 2 * \
                     np.exp(-a[10] * r ** 2) * \
                     (3 + a[10] * r ** 2 * (3 - 2 * a[10] * r ** 2))

                r = r - f / fp
                diff = np.abs(z - (0.27 * ppr / tpr / r)).max()
                z = 0.27 * ppr / tpr / r

            # gas compressibility from Dranchuk and Abau-Kassem
            cpr = tpr * z / ppr / fp
            cg = cpr / ppc

            # gas expansion factor
            bg = (0.0282793 * z * (form_temp + 459.67)) / pore_press

            # gas density
            rho_hc = 1.495 * 10 ** -3 * (pore_press * gas_grav) / \
                     (z * (form_temp + 459.67))
            nphi_hc = 2.17 * rho_hc

            # gas viscosity Lee Gonzalez Eakin method
            k = ((9.379 + 0.01607 * (28.9625 * gas_grav)) * (form_temp + 459.67) ** 1.5) / \
                (209.2 + 19.26 * (28.9625 * gas_grav) + (form_temp + 459.67))

            x = 3.448 + 986.4 / \
                (form_temp + 459.67) + 0.01009 * (28.9625 * gas_grav)

            y = 2.447 - 0.2224 * x
            mu_hc = 10 ** -4 * k * np.exp(x * rho_hc ** y)


        # oil reservoir
        else:

            # Normalize gas gravity to separator pressure of 100 psi
            ygs100 = gas_grav * (1 + 5.912 * 0.00001 * oil_api * (t_sep - 459.67) * np.log10(p_sep / 114.7))

            if oil_api < 30:
                if rs == 0 or rs is None:
                    rs = 0.0362 * ygs100 * pore_press ** 1.0937 * \
                         np.exp((25.724 * oil_api) / (form_temp + 459.67))

                bp = ((56.18 * rs / ygs100) * 10 ** (-10.393 * oil_api / (form_temp + 459.67))) ** 0.84246
                # gas saturated bubble-point
                bo = 1 + 4.677 * 10 ** -4 * rs + 1.751 * 10 ** -5 * \
                     (form_temp - 60) * (oil_api / ygs100) - \
                     1.811 * 10 ** -8 * rs * \
                     (form_temp - 60) * (oil_api / ygs100)
            else:
                if rs == 0 or rs is None:
                    rs = 0.0178 * ygs100 * pore_press ** 1.187 * \
                         np.exp((23.931 * oil_api) / (form_temp + 459.67))

                bp = ((56.18 * rs / ygs100) * 10 ** (-10.393 * oil_api / (form_temp + 459.67))) ** 0.84246

                # gas saturated bubble-point
                bo = 1 + 4.670 * 10 ** -4 * rs + 1.1 * \
                     10 ** -5 * (form_temp - 60) * (oil_api / ygs100) + \
                     1.337 * 10 ** -9 * rs * (form_temp - 60) * \
                     (oil_api / ygs100)

            # calculate bo for under saturated oil
            pp_gt_bp = np.where(pore_press > bp + 100)[0]
            if len(pp_gt_bp) > 0:
                bo[pp_gt_bp] = bo[pp_gt_bp] * np.exp(
                    -(0.00001 * (-1433 + 5 * rs + 17.2 * form_temp[pp_gt_bp] - 1180 * ygs100 + 12.61 * oil_api)) * \
                    np.log(pore_press[pp_gt_bp] / bp[pp_gt_bp]))

            # oil properties
            rho_hc = (((141.5 / (oil_api + 131.5) * 62.428) + 0.0136 * rs * ygs100) / bo) / 62.428
            nphi_hc = 1.003 * rho_hc

            # oil viscosity from Beggs-Robinson
            muod = 10 ** (np.exp(6.9824 - 0.04658 * oil_api) * form_temp ** -1.163) - 1

            mu_hc = (10.715 * (rs + 100) ** -0.515) * \
                    muod ** (5.44 * (rs + 150) ** -0.338)

            # under saturated oil viscosity from Vasquez and Beggs
            if len(pp_gt_bp) > 0:
                mu_hc[pp_gt_bp] = mu_hc[pp_gt_bp] * \
                                  (pore_press[pp_gt_bp] / bp[pp_gt_bp]) ** \
                                  (2.6 * pore_press[pp_gt_bp] ** 1.187 * 10 ** (-0.000039 * pore_press[pp_gt_bp] - 5))

        output_curves = [
            {'mnemonic': 'PORE_PRESS', 'data': pore_press, 'unit': 'psi',
             'descr': 'Calculated Pore Pressure'},

            {'mnemonic': 'RES_TEMP', 'data': form_temp, 'unit': 'F',
             'descr': 'Calculated Reservoir Temperature'},

            {'mnemonic': 'NES', 'data': nes, 'unit': 'psi',
             'descr': 'Calculated Net Effective Stress'},

            {'mnemonic': 'RW', 'data': rw, 'unit': 'ohmm',
             'descr': 'Calculated Resistivity Water'},

            {'mnemonic': 'RMF', 'data': rmf, 'unit': 'ohmm',
             'descr': 'Calculated Resistivity Mud Filtrate'},

            {'mnemonic': 'RHO_HC', 'data': rho_hc, 'unit': 'g/cc',
             'descr': 'Calculated Density of Hydrocarbon'},

            {'mnemonic': 'RHO_W', 'data': rho_w, 'unit': 'g/cc',
             'descr': 'Calculated Density of Water'},

            {'mnemonic': 'RHO_MF', 'data': rho_mf, 'unit': 'g/cc',
             'descr': 'Calculated Density of Mud Filtrate'},

            {'mnemonic': 'NPHI_HC', 'data': nphi_hc, 'unit': 'v/v',
             'descr': 'Calculated Neutron Log Response of Hydrocarbon'},

            {'mnemonic': 'NPHI_W', 'data': nphi_w, 'unit': 'v/v',
             'descr': 'Calculated Neutron Log Response of Water'},

            {'mnemonic': 'NPHI_MF', 'data': nphi_mf, 'unit': 'v/v',
             'descr': 'Calculated Neutron Log Response of Mud Filtrate'},

            {'mnemonic': 'MU_HC', 'data': mu_hc, 'unit': 'cP',
             'descr': 'Calculated Viscosity of Hydrocarbon'}
        ]

        for curve in output_curves:
            if curve['mnemonic'] in self.keys():
                self[curve['mnemonic']][depth_index] = curve['data']
            else:
                data = np.empty(len(self[0]))
                data[:] = np.nan
                data[depth_index] = curve['data']
                curve['data'] = data
                self.append_curve(curve['mnemonic'], data=curve['data'],
                                  unit=curve['unit'], descr=curve['descr'])

        # gas curves
        if oil_api == 0:
            gas_curves = [
                {'mnemonic': 'Z', 'data': z, 'unit': '',
                 'descr': 'Calculated Real Gas Z Factor'},

                {'mnemonic': 'CG', 'data': cg, 'unit': '1 / psi',
                 'descr': 'Calculated Gas Compressibility'},

                {'mnemonic': 'BG', 'data': bg, 'unit': '',
                 'descr': 'Calculated Gas Formation Volume Factor'}
            ]

            for curve in gas_curves:
                if curve['mnemonic'] in self.keys():
                    self[curve['mnemonic']][depth_index] = curve['data']
                else:
                    data = np.empty(len(self[0]))
                    data[:] = np.nan
                    data[depth_index] = curve['data']
                    curve['data'] = data
                    self.append_curve(curve['mnemonic'],
                                      data=curve['data'],
                                      unit=curve['unit'],
                                      descr=curve['descr'])

        # oil curves
        else:
            oil_curves = [
                {'mnemonic': 'BO', 'data': bo, 'unit': '',
                 'descr': 'Calculated Oil Formation Volume Factor'},

                {'mnemonic': 'BP', 'data': bp, 'unit': 'psi',
                 'descr': 'Calculated Bubble Point'}
            ]

            for curve in oil_curves:
                if curve['mnemonic'] in self.keys():
                    self[curve['mnemonic']][depth_index] = curve['data']
                else:
                    data = np.empty(len(self[0]))
                    data[:] = np.nan
                    data[depth_index] = curve['data']
                    curve['data'] = data
                    self.append_curve(curve['mnemonic'],
                                      data=curve['data'],
                                      unit=curve['unit'],
                                      descr=curve['descr'])

    def formation_fluid_properties(self, formations, parameter='default'):
        for form in formations:
            top = self.tops[form]
            bottom = self.formation_bottom_depth(form)
            params = self.fluid_properties_params[parameter]
            self.fluid_properties(top=top, bottom=bottom, **params)

    def load_multilateral_parameters(self, csv_path=None):
        """
        Reads parameters from a csv for input into the multimineral
        model.

        This method reads the file located at the csv_path and turns
        the values into dictionaries to be used as inputs into the
        multimineral method.
        """
        if csv_path is None:
            local_path = os.path.dirname(__file__)
            csv_path = os.path.join(local_path, 'data', 'multimineral.csv')

        df = pd.read_csv(csv_path)
        df = df.set_index('name')

        self.multimineral_parameters = df.to_dict(orient='index')

    def multimineral_model(self, top=None, bottom=None,
                           gr_matrix=10, nphi_matrix=0, gr_clay=350, rho_clay=2.64,
                           nphi_clay=0.65, pe_clay=4, rma=180, rt_clay=80,
                           vclay_linear_weight=1, vclay_clavier_weight=0.5,
                           vclay_larionov_weight=0.5, vclay_nphi_weight=1,
                           vclay_nphi_rhob_weight=1, vclay_cutoff=0.1, rho_om=1.15,
                           nphi_om=0.6, pe_om=0.2, ro=1.6, lang_press=670,
                           passey_nphi_weight=1, passey_rhob_weight=1, passey_lom=10,
                           passey_baseline_res=40, passey_baseline_rhob=2.65,
                           passey_baseline_nphi=0, schmoker_weight=1,
                           schmoker_slope=0.7257, schmoker_baseline_rhob=2.6,
                           rho_pyr=5, nphi_pyr=0.13, pe_pyr=13, om_pyrite_slope=0.2,
                           include_qtz='YES', rho_qtz=2.65, nphi_qtz=-0.04,
                           pe_qtz=1.81, include_clc='YES', rho_clc=1.71, nphi_clc=0,
                           pe_clc=5.08, include_dol='YES', rho_dol=2.85,
                           nphi_dol=0.04, pe_dol=3.14, include_x='NO',
                           name_x='Gypsum', name_log_x='GYP', rho_x=2.35,
                           nphi_x=0.507, pe_x=4.04, pe_fl=0, m=2, n=2, a=1,
                           archie_weight=0, indonesia_weight=1, simandoux_weight=0,
                           modified_simandoux_weight=0, waxman_smits_weight=0, cec=-1,
                           buckles_parameter=-1):

        # initialize required curves
        required_raw_curves = ['GR_N', 'NPHI_N', 'RHOB_N', 'RESDEEP_N']

        # check if PE is available
        if 'PE_N' in self.keys():
            use_pe = True
            required_raw_curves += ['PE_N']
        else:
            use_pe = False

        # check for requirements
        for curve in required_raw_curves:
            if curve not in self.keys():
                raise ValueError('Raw curve %s not found and is \
                             required for multimineral_model.' % curve)

        required_curves_from_fluid_properties = ['RW', 'RHO_HC',
                                                 'RHO_W', 'NPHI_HC',
                                                 'NPHI_W', 'RES_TEMP',
                                                 'NES', 'PORE_PRESS']

        for curve in required_curves_from_fluid_properties:
            if curve not in self.keys():
                raise ValueError('Fluid Properties curve %s not found.\
                     Run fluid_properties before multimineral_model.' \
                                 % curve)

        all_required_curves = required_raw_curves + \
                              required_curves_from_fluid_properties

        if 'BO' not in self.keys() and 'BG' not in self.keys():
            raise ValueError('Formation Volume Factor required for \
                      multimineral_model. Run fluid_properties first.')

        if 'BO' in self.keys():
            hc_class = 'OIL'
        else:
            hc_class = 'GAS'

        # initialize minerals
        if include_qtz.upper()[0] == 'Y':
            include_qtz = True
        else:
            include_qtz = False

        if include_clc.upper()[0] == 'Y':
            include_clc = True
        else:
            include_clc = False

        if include_dol.upper()[0] == 'Y':
            include_dol = True
        else:
            include_dol = False

        if include_x.upper()[0] == 'Y':
            include_x = True
            name_log_x = name_log_x.upper()
        else:
            include_x = False

        ## check for existence of calculated curves ###
        ### add if not found ##
        nulls = np.empty(len(self[0]))
        nulls[:] = np.nan

        output_curves = [
            {'mnemonic': 'PHIE', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Effective Porosity'},

            {'mnemonic': 'SW', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Water Saturation'},

            {'mnemonic': 'SHC', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Hydrocarbon Saturation'},

            {'mnemonic': 'BVH', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Bulk Volume Hydrocarbon'},

            {'mnemonic': 'BVW', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Bulk Volume Water'},

            {'mnemonic': 'BVWI', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Bulk Volume Water Irreducible'},

            {'mnemonic': 'BVWF', 'data': np.copy(nulls), 'unit':
                'v/v', 'descr': 'Bulk Volume Water Free'},

            {'mnemonic': 'BVOM', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Bulk Volume Fraction Organic Matter'},

            {'mnemonic': 'BVCLAY', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Bulk Volume Fraction Clay'},

            {'mnemonic': 'BVPYR', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Bulk Volume Fraction Pyrite'},

            {'mnemonic': 'VOM', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Matrix Volume Fraction Organic Matter'},

            {'mnemonic': 'VCLAY', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Matrix Volume Fraction Clay'},

            {'mnemonic': 'VPYR', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Matrix Volume Fraction Pyrite'},

            {'mnemonic': 'RHOM', 'data': np.copy(nulls), 'unit': 'g/cc',
             'descr': 'Matrix Density'},

            {'mnemonic': 'TOC', 'data': np.copy(nulls), 'unit': 'wt/wt',
             'descr': 'Matrix Weight Fraction Organic Matter'},

            {'mnemonic': 'WTCLAY', 'data': np.copy(nulls), 'unit': 'wt/wt',
             'descr': 'Matrix Weight Fraction Clay'},

            {'mnemonic': 'WTPYR', 'data': np.copy(nulls), 'unit': 'wt/wt',
             'descr': 'Matrix Weight Fraction Pyrite'},
        ]
        for curve in output_curves:
            if curve['mnemonic'] not in self.keys():
                self.append_curve(curve['mnemonic'], curve['data'],
                                  unit=curve['unit'],
                                  descr=curve['descr'])

        qtz_curves = [
            {'mnemonic': 'BVQTZ', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Bulk Volume Fraction Quartz'},
            {'mnemonic': 'VQTZ', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Matrix Volume Fraction Quartz'},
            {'mnemonic': 'WTQTZ', 'data': np.copy(nulls), 'unit': 'wt/wt',
             'descr': 'Matrix Weight Fraction Quartz'}
        ]
        if include_qtz:
            for curve in qtz_curves:
                if curve['mnemonic'] not in self.keys():
                    self.append_curve(curve['mnemonic'], curve['data'],
                                      unit=curve['unit'],
                                      descr=curve['descr'])

        clc_curves = [
            {'mnemonic': 'BVCLC', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Bulk Volume Fraction Calcite'},
            {'mnemonic': 'VCLC', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Matrix Volume Fraction Calcite'},
            {'mnemonic': 'WTCLC', 'data': np.copy(nulls), 'unit': 'wt/wt',
             'descr': 'Matrix Weight Fraction Calcite'}
        ]
        if include_clc:
            for curve in clc_curves:
                if curve['mnemonic'] not in self.keys():
                    self.append_curve(curve['mnemonic'], curve['data'],
                                      unit=curve['unit'],
                                      descr=curve['descr'])

        dol_curves = [
            {'mnemonic': 'BVDOL', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Bulk Volume Fraction Dolomite'},
            {'mnemonic': 'VDOL', 'data': np.copy(nulls), 'unit': 'v/v',
             'descr': 'Matrix Volume Fraction Dolomite'},
            {'mnemonic': 'WTDOL', 'data': np.copy(nulls), 'unit': 'wt/wt',
             'descr': 'Matrix Weight Fraction Dolomite'}
        ]
        if include_dol:
            for curve in dol_curves:
                if curve['mnemonic'] not in self.keys():
                    self.append_curve(curve['mnemonic'], curve['data'],
                                      unit=curve['unit'],
                                      descr=curve['descr'])

        min_x_curves = [
            {'mnemonic': 'V' + name_log_x, 'data': np.copy(nulls),
             'unit': 'v/v', 'descr': 'Bulk Volume Fraction ' + name_x},
            {'mnemonic': 'V' + name_log_x, 'data': np.copy(nulls),
             'unit': 'v/v', 'descr': 'Matrix Volume Fraction ' + name_x},
            {'mnemonic': 'WT' + name_log_x, 'data': np.copy(nulls),
             'unit': 'wt/wt', 'descr': 'Matrix Weight Fraction ' + name_x}
        ]
        if include_x:
            for curve in min_x_curves:
                if curve['mnemonic'] not in self.keys():
                    self.append_curve(curve['mnemonic'], curve['data'],
                                      unit=curve['unit'],
                                      descr=curve['descr'])

        oil_curve = {'mnemonic': 'OIP', 'data': np.copy(nulls),
                     'unit': 'Mmbbl / section', 'descr': 'Oil in Place'}
        if hc_class == 'OIL':
            if oil_curve['mnemonic'] not in self.keys():
                self.append_curve(oil_curve['mnemonic'], oil_curve['data'],
                                  unit=curve['unit'],
                                  descr=curve['descr'])

        gas_curves = [
            {'mnemonic': 'GIP', 'data': np.copy(nulls),
             'unit': 'BCF / section', 'descr': 'Gas in Place'},
            {'mnemonic': 'GIP_FREE', 'data': np.copy(nulls),
             'unit': 'BCF / section', 'descr': 'Free Gas in Place'},
            {'mnemonic': 'GIP_ADS', 'data': np.copy(nulls),
             'unit': 'BCF / section', 'descr': 'Adsorbed Gas in Place'}
        ]
        if hc_class == 'GAS':
            for curve in gas_curves:
                if curve['mnemonic'] not in self.keys():
                    self.append_curve(curve['mnemonic'], curve['data'],
                                      unit=curve['unit'],
                                      descr=curve['descr'])

        ### calculations over depths ###
        depth_index = np.intersect1d(np.where(self[0] >= top)[0],
                                     np.where(self[0] < bottom)[0])
        for i in depth_index:

            ### check for null values in data, skip if true ###
            nans = np.isnan([self[x][i] for x in all_required_curves])
            infs = np.isinf([self[x][i] for x in all_required_curves])
            if True in nans or True in infs: continue

            if i > 0:
                sample_rate = abs(self[0][i] - self[0][i - 1])
            else:
                sample_rate = abs(self[0][0] - self[0][1])

            ### initial parameters to start iterations ###
            phie = 0.1
            rhom = 2.68
            rho_fl = 1
            nphi_fl = 1
            vom = 0

            bvqtz_prev = 1
            bvclc_prev = 1
            bvdol_prev = 1
            bvx_prev = 1
            phi_prev = 1
            bvom_prev = 1
            bvclay_prev = 1
            bvpyr_prev = 1

            diff = 1
            counter = 0
            while diff > 1 * 10 ** -3 and counter < 20:
                counter += 1

                ### log curves without organics ###
                rhoba = self['RHOB_N'][i] + (rhom - rho_om) * vom
                nphia = self['NPHI_N'][i] + (nphi_matrix - nphi_om) * vom

                ### clay solver ###
                gr_index = np.clip((self['GR_N'][i] - gr_matrix) \
                                   / (gr_clay - gr_matrix), 0, 1)

                ### linear vclay method ###
                vclay_linear = gr_index

                ### Clavier vclay method ###
                vclay_clavier = np.clip(1.7 - np.sqrt(3.38 - \
                                                      (gr_index + 0.7) ** 2), 0, 1)

                ### larionov vclay method ###
                vclay_larionov = np.clip(0.083 * \
                                         (2 ** (3.7 * gr_index) - 1), 0, 1)

                # Neutron vclay method without organic correction
                vclay_nphi = np.clip((nphia - nphi_matrix) / \
                                     (nphi_clay - nphi_matrix), 0, 1)

                # Neutron Density vclay method with organic correction
                m1 = (nphi_fl - nphi_matrix) / (rho_fl - rhom)
                x1 = nphia + m1 * (rhom - rhoba)
                x2 = nphi_clay + m1 * (rhom - rho_clay)
                if x2 - nphi_matrix != 0:
                    vclay_nphi_rhob = np.clip((x1 - nphi_matrix) / \
                                              (x2 - nphi_matrix), 0, 1)
                else:
                    vclay_nphi_rhob = 0

                vclay_weights_sum = vclay_linear_weight + \
                                    vclay_clavier_weight + vclay_larionov_weight + \
                                    vclay_nphi_weight + vclay_nphi_rhob_weight

                vclay = (
                                vclay_linear_weight * vclay_linear + vclay_clavier_weight * vclay_clavier + vclay_larionov_weight * vclay_larionov + vclay_nphi_weight * vclay_nphi + vclay_nphi_rhob_weight * vclay_nphi_rhob) / \
                        vclay_weights_sum

                vclay = np.clip(vclay, 0, 1)

                bvclay = vclay * (1 - phie)

                ### organics ###
                if vclay > vclay_cutoff:

                    ### Passey ###
                    dlr_nphi = np.log10(self['RESDEEP_N'][i] / \
                                        passey_baseline_res) + 4 * (self['NPHI_N'][i] - passey_baseline_nphi)

                    dlr_rhob = np.log10(self['RESDEEP_N'][i] / \
                                        passey_baseline_res) - 2.5 * (self['RHOB_N'][i] - passey_baseline_rhob)

                    toc_nphi = np.clip((dlr_nphi * 10 ** (2.297 - 0.1688 * passey_lom) / 100), 0, 1)

                    toc_rhob = np.clip((dlr_rhob * 10 ** (2.297 - 0.1688 * passey_lom) / 100), 0, 1)

                    ### Schmoker ###
                    toc_sch = np.clip(schmoker_slope * \
                                      (schmoker_baseline_rhob - self['RHOB_N'][i]), 0, 1)

                    toc_weights = passey_nphi_weight + \
                                  passey_rhob_weight + schmoker_weight

                    ### toc in weight percent ###
                    toc = (
                                  passey_nphi_weight * toc_nphi + passey_rhob_weight * toc_rhob + schmoker_weight * toc_sch) / toc_weights

                    ### weight percent to volume percent ###
                    volume_om = toc / rho_om

                    # matrix density without organic matter
                    rhom_no_om = (rhom - toc * rho_om) / (1 - toc)

                    # volume of non-organics
                    volume_else = (1 - toc) / rhom_no_om

                    volume_total = volume_om + volume_else

                    vom = volume_om / volume_total
                    bvom = vom * (1 - phie)

                else:
                    toc = 0
                    vom = 0
                    bvom = 0

                ### pyrite correlation with organics ###
                vpyr = np.clip(om_pyrite_slope * vom, 0, 1)
                bvpyr = vpyr * (1 - phie)

                # create C, V, and L matrix for equations in Chapter 4 of
                # Principles of Mathematical Petrophysics by Doveton #

                # removed effect of clay, organics, and pyrite
                volume_unconventional = bvom + bvclay + bvpyr
                rhob_clean = (self['RHOB_N'][i] - (rho_om * bvom + rho_clay * bvclay + rho_pyr * bvpyr)) / \
                             (1 - volume_unconventional)

                nphi_clean = (self['NPHI_N'][i] - (nphi_om * bvom + nphi_clay * bvclay + nphi_pyr * bvpyr)) / \
                             (1 - volume_unconventional)

                minerals = []
                if use_pe:
                    pe_clean = (self['PE_N'][i] - (pe_om * bvom + pe_clay * bvclay + pe_pyr * bvpyr)) / \
                               (1 - bvom - bvclay - bvpyr)

                    l_clean = np.asarray([rhob_clean, nphi_clean,
                                          pe_clean, 1])

                    l = np.asarray([self['RHOB_N'][i],
                                    self['NPHI_N'][i],
                                    self['PE_N'][i], 1])

                    c_clean = np.asarray([0, 0, 0])  # initialize matrix C

                    if include_qtz:
                        minerals.append('QTZ')
                        mineral_matrix = np.asarray((rho_qtz, nphi_qtz,
                                                     pe_qtz))
                        c_clean = np.vstack((c_clean, mineral_matrix))

                    if include_clc:
                        minerals.append('CLC')
                        mineral_matrix = np.asarray((rho_clc, nphi_clc,
                                                     pe_clc))
                        c_clean = np.vstack((c_clean, mineral_matrix))

                    if include_dol:
                        minerals.append('DOL')
                        mineral_matrix = np.asarray((rho_dol, nphi_dol,
                                                     pe_dol))
                        c_clean = np.vstack((c_clean, mineral_matrix))

                    if include_x:
                        minerals.append('X')
                        mineral_matrix = np.asarray((rho_x, nphi_x,
                                                     pe_x))
                        c_clean = np.vstack((c_clean, mineral_matrix))

                    fluid_matrix = np.asarray((rho_fl, nphi_fl, pe_fl))
                    c_clean = np.vstack((c_clean, fluid_matrix))
                    minerals.append('PHI')

                else:
                    l_clean = np.asarray([rhob_clean, nphi_clean, 1])
                    l = np.asarray([self['RHOB_N'][i],
                                    self['NPHI_N'][i], 1])

                    c_clean = np.asarray((0, 0))  # initialize matrix C

                    if include_qtz:
                        minerals.append('QTZ')
                        mineral_matrix = np.asarray((rho_qtz, nphi_qtz))
                        c_clean = np.vstack((c_clean, mineral_matrix))

                    if include_clc:
                        minerals.append('CLC')
                        mineral_matrix = np.asarray((rho_clc, nphi_clc))
                        c_clean = np.vstack((c_clean, mineral_matrix))

                    if include_dol:
                        minerals.append('DOL')
                        mineral_matrix = np.asarray((rho_dol, nphi_dol))
                        c_clean = np.vstack((c_clean, mineral_matrix))

                    if include_x:
                        minerals.append('X')
                        mineral_matrix = np.asarray((rho_x, nphi_x))
                        c_clean = np.vstack((c_clean, mineral_matrix))

                    fluid_matrix = np.asarray((rho_fl, nphi_fl))
                    c_clean = np.vstack((c_clean, fluid_matrix))
                    minerals.append('PHI')

                c_clean = np.delete(c_clean, 0, 0)

                c_clean = np.vstack((c_clean.T,
                                     np.ones_like(c_clean.T[0])))

                bv_clean = nnls(c_clean, l_clean.T)[0]

                bvqtz = 0
                bvclc = 0
                bvdol = 0
                bvx = 0

                component_sum = np.sum(bv_clean)

                for s, mineral in enumerate(minerals):
                    if mineral == 'QTZ':
                        bvqtz = (bv_clean[s] / component_sum) * \
                                (1 - volume_unconventional)
                        bv_clean[s] = bvqtz
                    if mineral == 'CLC':
                        bvclc = (bv_clean[s] / component_sum) * \
                                (1 - volume_unconventional)
                        bv_clean[s] = bvclc
                    if mineral == 'DOL':
                        bvdol = (bv_clean[s] / component_sum) * \
                                (1 - volume_unconventional)
                        bv_clean[s] = bvdol
                    if mineral == 'X':
                        bvx = (bv_clean[s] / component_sum) * \
                              (1 - volume_unconventional)
                        bv_clean[s] = bvx
                    if mineral == 'PHI':
                        phie = (bv_clean[s] / component_sum) * \
                               (1 - volume_unconventional)
                        bv_clean[s] = phie

                if use_pe:
                    c = np.hstack((c_clean, np.asarray(
                        (
                            (rho_om, rho_clay, rho_pyr),
                            (nphi_om, nphi_clay, nphi_pyr),
                            (pe_om, pe_clay, pe_pyr),
                            (1, 1, 1)
                        )
                    )
                                   ))
                else:
                    c = np.hstack((c_clean, np.asarray(
                        (
                            (rho_om, rho_clay, rho_pyr),
                            (nphi_om, nphi_clay, nphi_pyr),
                            (1, 1, 1))
                    )
                                   ))

                bv = np.append(bv_clean, (bvom, bvclay, bvpyr))

                l_hat = np.dot(c, bv)

                sse = np.dot((l - l_hat).T, l - l_hat)

                prev = np.asarray((bvqtz_prev, bvclc_prev, bvdol_prev,
                                   bvx_prev, phi_prev, bvom_prev,
                                   bvclay_prev, bvpyr_prev))
                cur = np.asarray((bvqtz, bvclc, bvdol, bvx, phie, bvom,
                                  bvclay, bvpyr))

                diff = np.abs(cur - prev).sum()

                bvqtz_prev = bvqtz
                bvclc_prev = bvclc
                bvdol_prev = bvdol
                bvx_prev = bvx
                bvom_prev = bvom
                bvclay_prev = bvclay
                bvpyr_prev = bvpyr
                phi_prev = phie

                avg_percent_error = np.mean(np.abs(l - l_hat) / l) * 100

                # calculate matrix volume fraction

                per_matrix = 1 - phie

                vqtz = bvqtz / per_matrix
                vclc = bvclc / per_matrix
                vdol = bvdol / per_matrix
                vx = bvx / per_matrix
                vclay = bvclay / per_matrix
                vom = bvom / per_matrix
                vpyr = bvpyr / per_matrix

                # calculate weight fraction

                mass_qtz = vqtz * rho_qtz
                mass_clc = vclc * rho_clc
                mass_dol = vdol * rho_dol
                mass_x = vx * rho_x
                mass_om = vom * rho_om
                mass_clay = vclay * rho_clay
                mass_pyr = vpyr * rho_pyr

                rhom = mass_qtz + mass_clc + mass_dol + mass_x + \
                       mass_om + mass_clay + mass_pyr

                wtqtz = mass_qtz / rhom
                wtclc = mass_clc / rhom
                wtdol = mass_dol / rhom
                wtx = mass_x / rhom
                wtom = mass_om / rhom
                wtclay = mass_clay / rhom
                wtpyr = mass_pyr / rhom
                toc = wtom

                # saturations #

                # porosity cutoff in case phie =  0
                if phie < 0.001:
                    phis = 0.001
                else:
                    phis = phie

                # Archie
                sw_archie = np.clip(((a * self['RW'][i]) / (self['RESDEEP_N'][i] * (phis ** m))) ** (1 / n), 0, 1)

                ### Indonesia ###
                sw_ind_a = (phie ** m / self['RW'][i]) ** 0.5
                sw_ind_b = (vclay ** (2.0 - vclay) / rt_clay) ** 0.5
                sw_indonesia = np.clip(((sw_ind_a + sw_ind_b) ** 2.0 * self['RESDEEP_N'][i]) ** (-1 / n), 0, 1)

                ### Simandoux ###
                c = (1.0 - vclay) * a * self['RW'][i] / (phis ** m)
                d = c * vclay / (2.0 * rt_clay)
                e = c / self['RESDEEP_N'][i]
                sw_simandoux = np.clip(((d ** 2 + e) ** 0.2 - d) ** \
                                       (2 / n), 0, 1)

                ### modified Simandoux ###
                sw_mod_simd = np.clip((0.5 * self['RW'][i] / phis ** m) * (
                        (4 * phis ** m) / (self['RW'][i] * self['RESDEEP_N'][i]) + (vclay / rt_clay) ** 2) ** (
                                              1 / n) - \
                                      vclay / rt_clay, 0, 1)

                ### Waxman Smits ###
                if cec <= 0:
                    cec = 10 ** (1.9832 * vclay - 2.4473)

                rw77 = self['RESDEEP_N'][i] * (self['RES_TEMP'][i] + 6.8) \
                       / 83.8

                b = 4.6 * (1 - 0.6 * np.exp(-0.77 / rw77))
                f = a / (phis ** m)
                qv = cec * (1 - phis) * rhom / phis
                sw_waxman_smits = np.clip(0.5 * ((-b * qv * rw77) + ((b * qv * rw77) ** 2 + 4 * f * self['RW'][i] /
                                                                     self['RESDEEP_N'][i]) ** 0.5) \
                                          ** (2 / n), 0, 1)

                ### weighted calculation with bv output ###
                weight_saturations = archie_weight + indonesia_weight + \
                                     simandoux_weight + modified_simandoux_weight + \
                                     waxman_smits_weight

                sw = (
                             archie_weight * sw_archie + indonesia_weight * sw_indonesia + simandoux_weight * sw_simandoux + modified_simandoux_weight * sw_mod_simd + waxman_smits_weight * sw_waxman_smits) / \
                     weight_saturations

                bvw = phie * sw
                bvh = phie * (1 - sw)

                if hc_class == 'OIL':
                    oip = (7758 * 640 * sample_rate * bvh * 10 ** -6) / \
                          self['BO'][i]  # Mmbbl per sample rate

                elif hc_class == 'GAS':
                    langslope = (-0.08 * self['RES_TEMP'][i] + 2 * ro + 22.75) / 2
                    gas_ads = langslope * vom * 100 * \
                              (self['PORE_PRESS'][i] / (self['PORE_PRESS'][i] + lang_press))

                    gip_free = (43560 * 640 * sample_rate * bvh * 10 ** -9) \
                               / self['BG'][i]  # BCF per sample rate
                    gip_ads = (1359.7 * 640 * sample_rate * self['RHOB_N'][i] * gas_ads * 10 ** -9) / \
                              self['BG'][i]  # BCF per sample rate
                    gip = gip_free + gip_ads

                rho_fl = self['RHO_W'][i] * sw + \
                         self['RHO_HC'][i] * (1 - sw)

                nphi_fl = self['NPHI_W'][i] * sw + \
                          self['NPHI_HC'][i] * (1 - sw)

            ### save calculations to log ###

            ### bulk volume ###
            self['BVOM'][i] = bvom
            self['BVCLAY'][i] = bvclay
            self['BVPYR'][i] = bvpyr

            if include_qtz:
                self['BVQTZ'][i] = bvqtz
            if include_clc:
                self['BVCLC'][i] = bvclc
            if include_dol:
                self['BVDOL'][i] = bvdol
            if include_x:
                self['BV' + name_log_x][i] = bvx

            self['BVH'][i] = bvh
            self['BVW'][i] = bvw

            ### porosity and saturations ###
            self['PHIE'][i] = phie
            self['SW'][i] = sw
            self['SHC'][i] = 1 - sw

            ### mineral volumes ###
            self['VOM'][i] = vom
            self['VCLAY'][i] = vclay
            self['VPYR'][i] = vpyr

            if include_qtz:
                self['VQTZ'][i] = vqtz
            if include_clc:
                self['VCLC'][i] = vclc
            if include_dol:
                self['VDOL'][i] = vdol
            if include_x:
                self['V' + name_log_x] = vx

            ### weight percent ###
            self['RHOM'][i] = rhom
            self['TOC'][i] = toc
            self['WTCLAY'][i] = wtclay
            self['WTPYR'][i] = wtpyr

            if include_qtz:
                self['WTQTZ'][i] = wtqtz
            if include_clc:
                self['WTCLC'][i] = wtclc
            if include_dol:
                self['WTDOL'][i] = wtdol
            if include_x:
                self['WT' + name_log_x] = wtx

            # find irreducible water if buckles_parameter is specified
            if buckles_parameter > 0:
                sw_irr = buckles_parameter / (phie / (1 - vclay))
                bvwi = phie * sw_irr
                bvwf = bvw - bvwi
                self['BVWI'][i] = bvwi
                self['BVWF'][i] = bvwf

            if hc_class == 'OIL':
                self['OIP'][i] = oip

            elif hc_class == 'GAS':
                self['GIP_FREE'][i] = gip_free
                self['GIP_ADS'][i] = gip_ads
                self['GIP'][i] = gip

        ### find irreducible water saturation outside of loop ###
        ### since parameters depend on calculated values ###

        if buckles_parameter < 0:
            buckles_parameter = np.mean(self['PHIE'][depth_index] * \
                                        self['SW'][depth_index])

            ir_denom = (self['PHIE'][depth_index] / (1 - self['VCLAY'][depth_index]))
            ir_denom[np.where(ir_denom < 0.001)[0]] = 0.001
            sw_irr = buckles_parameter / ir_denom

            self['BVWI'][depth_index] = \
                self['PHIE'][depth_index] * sw_irr

            self['BVWF'][depth_index] = self['BVW'][depth_index] - \
                                        self['BVWI'][depth_index]

    def export_csv(self, filepath=None, fill_null=True, **kwargs):
        """
        Export data to a csv file.

        Data will be exported as a 2-D DataFrame array, so that it keeps only self.keys() and self.data().
        Auto fill Null values are optional which uses null value from the LASFile. Set to 'True' by default.

        """
        if filepath is None:
            local_path = os.path.dirname(__file__)
            filepath = os.path.join(local_path, f"data/{self.filename}_las2csv.csv")
        if check_file(filepath):
            df = self.df()
            if fill_null:
                df.fillna(value=self.well['NULL'].value, inplace=True)
            df.to_csv(filepath, **kwargs)
        else:
            print('Export aborted')

    def export_excel(self, filepath=None):
        """
        Export data to an Excel file. Works differently than export_csv.

        This method export .xlsx file using superclass LASFile.to_excel function to convert
        LAS files into Excel, while retaining the header information.

        Output is a two-sheet Excel file stores header information in Sheet1 and data in Sheet2.

        """
        if filepath is None:
            local_path = os.path.dirname(__file__)
            filepath = os.path.join(local_path, f"data/{self.filename}_las2excel.xlsx")
        if check_file(filepath):
            self.to_excel(filepath)
        else:
            print('Export aborted')

    def write(self, filepath, version=2.0, wrap=False,
              STRT=None, STOP=None, STEP=None, fmt='%10.5g'):
        """
        Writes to las file, and overwrites if file exists. Uses parent
        class LASFile.write method with specified defaults.
        """
        if filepath is None:
            local_path = os.path.dirname(__file__)
            filepath = os.path.join(local_path, f"data/{self.filename}_edited.las")
        with open(filepath, 'w') as f:
            super(Log, self).write(f, version=version, wrap=wrap,
                                   STRT=STRT, STOP=STOP,
                                   STEP=None, fmt=fmt)
