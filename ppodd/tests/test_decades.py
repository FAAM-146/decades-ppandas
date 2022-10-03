import datetime
import os
import unittest
import logging

import pandas as pd
import ppodd.readers as r
from ppodd.decades import DecadesDataset, DecadesVariable, DecadesFile
from ppodd.decades.attributes import NonStandardAttributeError

logging.disable(logging.CRITICAL)

TEST_VAR_NAME = 'testvar'
TEST_VAR_VALUES = [1,2,3,4,5,6,7,8,9,10]
TEST_FILE_PATTERNS = {
    'CORCON01_20010101_000000_c123': r.TcpFileReader,
    'somefile.zip': r.ZipFileReader,
    'core_faam_20010101_v005_r0_c123.nc': r.CoreNetCDFReader,
    'GINDAT01_20010101_000000.bin': r.GinFileReader,
    'CORCON01_TCP_v1.csv': r.CrioDefinitionReader,
    'flight-cst_faam_20010101_r0_c123.yaml': r.YamlConstantsReader,
    'flight-cst_faam_20010101_r0_c123.json': r.JsonConstantsReader,
    'SEAPROBE_20010101_000000_c123.wcm': r.WcmFileReader,
    'faam-fgga_faam_20010101_r0_c123.na': r.FGGAReader,
    'CCN-200 data 010101010000.csv': r.CCNReader,
    'TWBOZO01_20000101_120130_C123.txt': r.ChemistryTxtReader,
    'CHTSOO02_20000101_120130_C123.txt': r.ChemistryTxtReader,
    'AL55CO01_20000101_120130_C123.txt': r.ChemistryTxtReader
}


class TestDecades(unittest.TestCase):
    """
    An empty subclass of unittest.TestCase, to which we append tests for each
    module.
    """

    def setUp(self):
        self.test_index_1 = pd.date_range(
            start=datetime.datetime(2000, 1, 1),
            periods=10, freq='S'
        )

        self.test_index_32 = pd.date_range(
            start=datetime.datetime(2000, 1, 1),
            periods=320, freq='31250000N'
        )

        self.d = DecadesDataset(datetime.datetime(2000, 1, 1))
        self.v = DecadesVariable(
            {TEST_VAR_NAME: TEST_VAR_VALUES}, index=self.test_index_1,
            frequency=1
        )

    @classmethod
    def setUpClass(cls):
        for pattern in TEST_FILE_PATTERNS.keys():
            with open(pattern, 'w'):
                pass

    @classmethod
    def teardownClass(cls):
        for pattern in TEST_FILE_PATTERNS.keys():
            os.remove(pattern)

    def _get_var_1(self):
        return  DecadesVariable({
            TEST_VAR_NAME: TEST_VAR_VALUES
        }, index=self.test_index_1)

    def _get_var_32(self):
        return  DecadesVariable({
            TEST_VAR_NAME: TEST_VAR_VALUES*32
        }, index=self.test_index_32)

    def test_create_dataset(self):
        d = DecadesDataset()

    def test_variable_isnumeric(self):
        self.assertTrue(self.v.isnumeric)

        v2 = DecadesVariable(
            {'stringvar': ['a'] * len(TEST_VAR_VALUES)},
            index=self.test_index_1, frequency=1
        )
        self.assertFalse(v2.isnumeric)

    def test_create_dataset_with_date(self):
        d = DecadesDataset(datetime.date.today())

    def test_create_decades_variable(self):
        self._get_var_1()

    def test_t0_correct(self):
        v = self._get_var_1()
        self.assertEqual(v.t0, self.test_index_1[0])

    def test_t1_correct(self):
        v = self._get_var_1()
        self.assertEqual(v.t1, self.test_index_1[-1])

    def test_time_bounds(self):
        v = self._get_var_1()
        i = self.test_index_1
        bounds = v.time_bounds()
        self.assertEqual(bounds[0], self.test_index_1[0])
        self.assertEqual(bounds[1], self.test_index_1[-1])

    def test_dataset_date_attribute(self):
        today = datetime.date.today()
        d = DecadesDataset(today)
        self.assertEqual(d.date, today)

    def test_calling_variable_name(self):
        v = self._get_var_1()()
        self.assertEqual(v.name, TEST_VAR_NAME)

    def test_calling_variable_index(self):
        v = self._get_var_1()()
        for a, b in zip(v.index, self.test_index_1):
            self.assertEqual(a, b)

    def test_get_freq_for_1hz_and_32hz(self):
        var1 = self._get_var_1()
        var32 = self._get_var_32()
        self.assertEqual(var1.frequency, 1)
        self.assertEqual(var32.frequency, 32)

    def test_calling_variable_values(self):
        v = self._get_var_1()()
        for a, b in zip(v.values, TEST_VAR_VALUES):
            self.assertEqual(a, b)

    def test_variable_array_attribute(self):
        v = self._get_var_1()
        for a, b in zip(v.array, TEST_VAR_VALUES):
            self.assertEqual(a, b)

    def test_variable_length(self):
        v = self._get_var_1()
        self.assertEqual(len(v), len(v.array))

    def test_variable_trim(self):
        v = self._get_var_1()
        start = self.test_index_1[3]
        end = self.test_index_1[-3]
        v.trim(start, end)
        self.assertEqual(v.t0, start)
        self.assertEqual(v.t1, end)
        f = v.flag()
        self.assertEqual(f.index[0], start)
        self.assertEqual(f.index[-1], end)

    def test_recover_index(self):
        v = self._get_var_1()
        index = v.index
        self.d.add_input(v)
        for a, b in zip(self.d[v.name].index, index):
            self.assertEqual(a, b)

    def test_variable_merge_contiguous(self):
        index1 = pd.date_range(
            start=datetime.datetime(2000, 1, 1),
            periods=10, freq='S'
        )

        index2 = pd.date_range(
            start=datetime.datetime(2000, 1, 1, 0, 0, 10),
            periods=10, freq='S'
        )

        v1 = DecadesVariable(
            {TEST_VAR_NAME: TEST_VAR_VALUES}, index=index1, frequency=1
        )

        v2 = DecadesVariable(
            {TEST_VAR_NAME: TEST_VAR_VALUES}, index=index2, frequency=1
        )

        v1.merge(v2)

        self.assertEqual(v1.t0, index1[0])
        self.assertEqual(v1.t1, index2[-1])
        self.assertEqual(len(v1), 2*(len(TEST_VAR_VALUES)))

    def test_variable_merge_noncontiguous(self):
        index1 = pd.date_range(
            start=datetime.datetime(2000, 1, 1),
            periods=10, freq='S'
        )

        index2 = pd.date_range(
            start=datetime.datetime(2000, 1, 1, 0, 0, 15),
            periods=10, freq='S'
        )

        v1 = DecadesVariable(
            {TEST_VAR_NAME: TEST_VAR_VALUES}, index=index1, frequency=1
        )

        v2 = DecadesVariable(
            {TEST_VAR_NAME: TEST_VAR_VALUES}, index=index2, frequency=1
        )

        v1.merge(v2)

        self.assertEqual(v1.t0, index1[0])
        self.assertEqual(v1.t1, index2[-1])
        self.assertEqual(len(v1), 2*(len(TEST_VAR_VALUES))+5)

    def test_variable_merge_interleaved(self):
        index1 = pd.date_range(
            start=datetime.datetime(2000, 1, 1),
            periods=10, freq='2S'
        )

        index2 = pd.date_range(
            start=datetime.datetime(2000, 1, 1, 0, 0, 1),
            periods=10, freq='2S'
        )

        v1 = DecadesVariable(
            {TEST_VAR_NAME: TEST_VAR_VALUES}, index=index1, frequency=1
        )

        v2 = DecadesVariable(
            {TEST_VAR_NAME: TEST_VAR_VALUES}, index=index2, frequency=1
        )

        v1.merge(v2)

        self.assertEqual(v1.t0, index1[0])
        self.assertEqual(v1.t1, index2[-1])
        self.assertEqual(len(v1), 2*(len(TEST_VAR_VALUES)))

        expected_array = []
        for i in TEST_VAR_VALUES:
            expected_array.append(i)
            expected_array.append(i)

        for a, b in zip(v1.array, expected_array):
            self.assertEqual(a, b)

    def test_infer_readers(self):
        for pattern, reader in TEST_FILE_PATTERNS.items():

            self.assertIs(
                DecadesDataset.infer_reader(DecadesFile(pattern)), reader
            )

            os.remove(pattern)

    def test_files_list(self):
        patterns = list(TEST_FILE_PATTERNS.keys())
        d = DecadesDataset()
        for pattern in patterns:
            d.add_file(pattern)

        for pattern in patterns:
            self.assertIn(pattern, [i.filepath for i in d.files])

    def test_add_decades_file(self):
        patterns = list(TEST_FILE_PATTERNS.keys())

        d = DecadesDataset()
        for pattern in patterns:
            d.add_file(pattern)

        files = []
        for i in d.readers:
            files += i.files

        for _file in files:
            self.assertIn(_file.filepath, patterns)

    def test_takeoff_land_time(self):
        vals = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
        prtaft = DecadesVariable(
            {'PRTAFT_wow_flag': vals}, index=self.test_index_1
        )
        self.d.add_input(prtaft)

        self.assertEqual(self.d.takeoff_time, self.test_index_1[4])
        self.assertEqual(self.d.landing_time, self.test_index_1[7])

    def test_dataset_time_bounds_inputs(self):
        v1 = DecadesVariable({'a': TEST_VAR_VALUES}, index=self.test_index_1)
        index2 = self.test_index_1 + datetime.timedelta(minutes=1)
        v2 = DecadesVariable({'b': TEST_VAR_VALUES}, index=index2)
        self.d.add_input(v1)
        self.d.add_input(v2)
        bnds = self.d.time_bounds()
        self.assertEqual(bnds[0], self.test_index_1[0])
        self.assertEqual(bnds[1], index2[-1])

    def test_dataset_time_bounds_outputs(self):
        v1 = DecadesVariable(
            {'a': TEST_VAR_VALUES}, index=self.test_index_1, frequency=1
        )
        index2 = self.test_index_1 + datetime.timedelta(minutes=1)
        v2 = DecadesVariable({'b': TEST_VAR_VALUES}, index=index2)
        self.d.add_output(v1)
        self.d.add_output(v2)
        bnds = self.d.time_bounds()
        self.assertEqual(bnds[0], self.test_index_1[0])
        self.assertEqual(bnds[1], index2[-1])

    def test_dataset_time_bounds_input_output(self):
        v1 = DecadesVariable(
            {'a': TEST_VAR_VALUES}, index=self.test_index_1, frequency=1
        )
        index2 = self.test_index_1 + datetime.timedelta(minutes=1)
        v2 = DecadesVariable(
            {'b': TEST_VAR_VALUES}, index=index2, frequency=1
        )
        self.d.add_output(v1)
        self.d.add_input(v2)
        bnds = self.d.time_bounds()
        self.assertEqual(bnds[0], self.test_index_1[0])
        self.assertEqual(bnds[1], index2[-1])

    def test_add_input_variable(self):
        self.d.add_input(self.v)
        self.assertIn(self.v, self.d._backend.inputs)

    def test_add_output_variable(self):
        self.d.add_output(self.v)
        self.assertIn(self.v, self.d._backend.outputs)

    def test_list_variables(self):
        v1 = DecadesVariable(
            {'a': TEST_VAR_VALUES}, index=self.test_index_1, frequency=1
        )
        v2 = DecadesVariable(
            {'b': TEST_VAR_VALUES}, index=self.test_index_1, frequency=1
        )

        self.d.add_input(v1)
        self.d.add_output(v2)

        self.assertIn(v1.name, self.d.variables)
        self.assertIn(v2.name, self.d.variables)

    def test_remove_variable(self):
        self.d.add_input(self.v)
        self.d.remove(self.v.name)
        self.assertEqual(self.d.variables, [])

        self.d.add_output(self.v)
        self.d.remove(self.v.name)
        self.assertEqual(self.d.variables, [])

    def test_add_constant(self):
        self.d.add_constant('LTUAE', 42)
        self.assertEqual(self.d['LTUAE'], 42)

    def test_lazy_constant_item(self):
        value = self.d.lazy['LTUAE']
        self.d.add_constant('LTUAE', 42)
        self.assertEqual(value(), 42)
        self.assertEqual(self.d.lazy['LTUAE'], 42)

    def test_outputs_list(self):
        v1 = self.v
        v2 = DecadesVariable({'v2': TEST_VAR_VALUES}, index=self.test_index_1,
                             frequency=1)
        self.d.add_output(v1)
        self.d.add_output(v2)
        self.assertIn(v1, self.d.outputs)
        self.assertIn(v2, self.d.outputs)

    def test_clear_outputs(self):
        v1 = self.v
        v2 = DecadesVariable({'v2': TEST_VAR_VALUES}, index=self.test_index_1,
                             frequency=1)
        self.d.add_output(v1)
        self.d.add_output(v2)
        self.d.clear_outputs()
        self.assertEqual(self.d.outputs, [])

    def test_add_global(self):
        self.assertRaises(
            NonStandardAttributeError, lambda: self.d.add_global('LTUAE', 42)
        )
        self.d.globals.strict = False
        self.d.add_global('LTUAE', 42)
        self.assertEqual(self.d.globals()['LTUAE'], 42)

    def test_add_interpolated_global(self):
        self.d.globals.strict = False
        self.d.add_global('one', 2)
        self.d.add_global('two', '{one}')
        self.d._interpolate_globals()
        self.assertEqual(self.d.globals()['two'], '2')

    def test_add_data_global(self):
        self.d.add_output(self.v)
        self.d.globals.strict = False
        self.d.add_global('test1', f'<data {TEST_VAR_NAME} max>')
        self.assertEqual(self.d.globals()['test1'], max(TEST_VAR_VALUES))
        self.d.add_global('test2', f'<call datetime.date.today>')
        self.assertEqual(self.d.globals()['test2'], datetime.date.today())
