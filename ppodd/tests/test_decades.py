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
    'TWBOZO01_20000101_120130_C123.txt': r.GenericTxtReader,
    'CHTSOO02_20000101_120130_C123.txt': r.GenericTxtReader,
    'AL55CO01_20000101_120130_C123.txt': r.GenericTxtReader,
    'ZEUS0001_20000101_120130_C123.txt': r.GenericTxtReader
}

def get_dataset():
    return DecadesDataset(datetime.datetime(2000, 1, 1))

def get_variable(name=TEST_VAR_NAME, values=TEST_VAR_VALUES, index=None):
    if index is None:
        index = pd.date_range(
            start=datetime.datetime(2000, 1, 1),
            periods=10, freq='S'
        )
    
    
    return DecadesVariable(
            {TEST_VAR_NAME: TEST_VAR_VALUES}, index=index,
            frequency=1
    )


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
        v = get_variable()
        self.assertTrue(v.isnumeric)

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
        d = get_dataset()
        index = v.index
        d.add_input(v)
        for a, b in zip(d[v.name].index, index): # type: ignore
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
            self.assertIn(pattern, [i.filepath for i in d.files]) # type: ignore

    def test_add_decades_file(self):
        patterns = list(TEST_FILE_PATTERNS.keys())

        d = DecadesDataset()
        for pattern in patterns:
            d.add_file(pattern)

        files = []
        for i in d.readers: # type: ignore
            files += i.files

        for _file in files:
            self.assertIn(_file.filepath, patterns)

    def test_takeoff_land_time(self):
        vals = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
        prtaft = DecadesVariable(
            {'PRTAFT_wow_flag': vals}, index=self.test_index_1
        )
        d = get_dataset()
        d.add_input(prtaft)

        self.assertEqual(d.takeoff_time, self.test_index_1[4])
        self.assertEqual(d.landing_time, self.test_index_1[7])

    def test_dataset_time_bounds_inputs(self):
        d = get_dataset()
        v1 = DecadesVariable({'a': TEST_VAR_VALUES}, index=self.test_index_1)
        index2 = self.test_index_1 + datetime.timedelta(minutes=1)
        v2 = DecadesVariable({'b': TEST_VAR_VALUES}, index=index2)
        d.add_input(v1)
        d.add_input(v2)
        bnds = d.time_bounds()
        self.assertEqual(bnds[0], self.test_index_1[0])
        self.assertEqual(bnds[1], index2[-1])

    def test_dataset_time_bounds_outputs(self):
        d = get_dataset()
        v1 = DecadesVariable(
            {'a': TEST_VAR_VALUES}, index=self.test_index_1, frequency=1
        )
        index2 = self.test_index_1 + datetime.timedelta(minutes=1)
        v2 = DecadesVariable({'b': TEST_VAR_VALUES}, index=index2)
        d.add_output(v1)
        d.add_output(v2)
        bnds = d.time_bounds()
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
        d = get_dataset()
        d.add_output(v1)
        d.add_input(v2)
        bnds = d.time_bounds()
        self.assertEqual(bnds[0], self.test_index_1[0])
        self.assertEqual(bnds[1], index2[-1])

    def test_add_input_variable(self):
        d = get_dataset()
        v = get_variable()
        d.add_input(v)
        self.assertIn(v, d._backend.inputs)

    def test_add_output_variable(self):
        d = get_dataset()
        v = get_variable()
        d.add_output(v)
        self.assertIn(v, d._backend.outputs)

    def test_list_variables(self):
        d = get_dataset()
        v1 = DecadesVariable(
            {'a': TEST_VAR_VALUES}, index=self.test_index_1, frequency=1
        )
        v2 = DecadesVariable(
            {'b': TEST_VAR_VALUES}, index=self.test_index_1, frequency=1
        )

        d.add_input(v1)
        d.add_output(v2)

        self.assertIn(v1.name, d.variables)
        self.assertIn(v2.name, d.variables)

    def test_remove_variable(self):
        d = get_dataset()
        v = DecadesVariable(
            {'a': TEST_VAR_VALUES}, index=self.test_index_1, frequency=1
        )

        d.add_input(v)
        d.remove(v.name)
        self.assertEqual(d.variables, [])

        d.add_output(v)
        d.remove(v.name)
        self.assertEqual(d.variables, [])

    def test_add_constant(self):
        d = get_dataset()
        d.add_constant('LTUAE', 42)
        self.assertEqual(d['LTUAE'], 42)

    def test_lazy_constant_item(self):
        d = get_dataset()
        value = d.lazy['LTUAE']
        d.add_constant('LTUAE', 42)
        self.assertEqual(value(), 42)
        self.assertEqual(d.lazy['LTUAE'], 42)

    def test_outputs_list(self):
        d = get_dataset()
        v1 = get_variable()
        v2 = DecadesVariable({'v2': TEST_VAR_VALUES}, index=self.test_index_1,
                             frequency=1)
        d.add_output(v1)
        d.add_output(v2)
        self.assertIn(v1, d.outputs)
        self.assertIn(v2, d.outputs)

    def test_clear_outputs(self):
        d = get_dataset()
        v1 = get_variable()
        v2 = DecadesVariable({'v2': TEST_VAR_VALUES}, index=self.test_index_1,
                             frequency=1)
        d.add_output(v1)
        d.add_output(v2)
        d.clear_outputs()
        self.assertEqual(d.outputs, [])

    def test_add_global(self):
        d = get_dataset()
        self.assertRaises(
            NonStandardAttributeError, lambda: d.add_global('LTUAE', 42)
        )
        d.globals.strict = False
        d.add_global('LTUAE', 42)
        self.assertEqual(d.globals()['LTUAE'], 42)

    def test_add_interpolated_global(self):
        d = get_dataset()
        d.globals.strict = False
        d.add_global('one', 2)
        d.add_global('two', '{one}')
        d._interpolate_globals()
        self.assertEqual(d.globals()['two'], '2')

    def test_add_data_global(self):
        d = get_dataset()
        v = get_variable()
        d.add_output(v)
        d.globals.strict = False
        d.add_global('test1', f'<data {TEST_VAR_NAME} max>')
        self.assertEqual(d.globals()['test1'], max(TEST_VAR_VALUES))
        d.add_global('test2', f'<call datetime.date.today>')
        self.assertEqual(d.globals()['test2'], datetime.date.today())
