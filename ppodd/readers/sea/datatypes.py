def striter_combine(s1, s2):
    return [_s1 + _s2 for _s1 in s1 for _s2 in s2]

# Define structure for identifier d0
d0_types = ['S2'] + ['S3', 'float32', 'float32', 'float32'] * 5
d0_names = ['id'] + striter_combine(
    ['elTWC', 'el083', 'el021', 'elCMP', 'elDCE'],
    ['', '_V', '_A', '_T']
)
d0_longnames = ['id'] + striter_combine(
    ['element TWC', 'element 083', 'element 021', 'element CMP', 'element DCE'],
    ['', ' voltage', ' current', ' temperature']
)
d0_units = [''] + ['', 'volt', 'amp', 'deg C'] * 5

# Define structure fo identifier d1
d1_types = ['S2'] + ['S3', 'float32'] * 3
d1_names = ['id'] + striter_combine(['elTWC', 'el083', 'el021'], ['', '_rtwc'])
d1_longnames = ['id', '', 'total water content', '', 'liquid water content',
                '', 'liquid water content']
d1_units = ['', '', 'g/m^3', '', 'g/m^3', '', 'g/m^3']

# Define structure fo identifier d2
d2_types = ['S2'] + ['S3', 'S6', 'int', 'i1', 'i1', 'i1'] * 5
d2_names = ['id'] + striter_combine(['elTWC', 'el083', 'el021'], ['', '_rtwc'])
d2_longnames = ['id'] + striter_combine(
    ['element TWC', 'element 083', 'element 021', 'element CMP', 'element DCE'],
    ['', '_status', '_DAC', '_pt', '_it', '_dt']
)

d2_units = [''] * 31

# Define structure fo identifier d3
d3_types = ['S2', 'object', 'object', 'float32', 'float32', 'float32', 'int',
            'float32']
d3_names = ['id', 'date', 'time', 'tas', 'tstatic', 'pstatic',
            'zerostate','powerboxtemp']
d3_longnames = ['id', 'date', 'time', 'true air speed', 'static temperature',
                'static pressure', 'request zero state',
                'internal temperature power box']
d3_units = ['', '', 'UTC', 'm/s', 'deg C', 'mb', '', 'deg C']

# Define structure fo identifier c0
c0_types = (
    ['S2', 'S4'] +
    ['S3', 'float32', 'float32', 'float32', 'float32', 'float32'] * 3 +
    ['S3', 'float32', 'float32', 'float32']
)

c0_names = ['id', 'sn'] + striter_combine(
    ['elTWC', 'el083', 'el021'], ['','_l','_w','_f','_s','_o']
) + striter_combine(['elCMP'],  ['','_l','_w','_f'])

c0_longnames = ['id', 'serial number'] + striter_combine(
    ['element TWC', 'element 083', 'element 021'],
    ['',' length',' width', ' fixture resistance', ' slope correction',
     'offset correction']
) + striter_combine(
    ['element CMP'], ['',' length',' width',' fixture resistance']
)

c0_units = (
    ['', ''] + ['', 'mm', 'mm', 'milliohm', '', ''] * 3
             + ['', 'mm', 'mm', 'milliohm']
)

# Define structure fo identifier cpbx
cpbx_types = (
    ['S2', 'S4'] +
    ['S3', 'float32', 'float32', 'float32', 'float32', 'float32'] * 3 +
    ['S3', 'float32', 'float32', 'float32']
)

cpbx_names = (
    ['id', 'sn', 'chipid', 'tid', 'endid', 'rev'] +
    striter_combine(
        ['ele1', 'ele2', 'ele3', 'cmp', 'dce'],
        ['state', 'vrawv', 'vrawi', 'shunt', 'maxamps', 'maxvolts', 'hardver',
         'softver']
    )
)

cpbx_longnames = (
    ['id', 'serial number', 'chip id', 'Temperature EPROM id', 'End EPROM id'],
    striter_combine(
        ['ele1', 'ele2', 'ele3', 'cmp', 'dce'],
        [' state of element', ' Vraw-V value', ' Vraw-I value', ' shunt',
         ' maximum amps', ' maximum volts', ' hardware Version',
         ' software versionr']
    )
)

cpbx_units = (
    ['',] * 6 + ['', 'volt', 'amp', 'milliohm', 'amp', 'volt', '', ''] * 5
)

# Define structure fo identifier cprb
cprb_types = (
    ['S4', 'S4', 'S16', 'S4'], ['S3'] + ['int'] * 5 + ['float32'] * 5 +
    (
        ['S6', 'float32', 'float32', 'float32', 'S3'] + ['int'] * 5 +
        ['float32'] * 5
    ) * 2 + ['S4', 'float32', 'float32', 'float32'] +
    ['S3'] + ['int'] * 5 + ['float32'] * 5 + ['S4', 'float32'] + ['S3'] +
    ['int']  * 11
)
cprb_names = (
    ['id','sn','chipid','rev'] +
    striter_combine(
        ['elTWC','el083','el021','elCMP','elDCE'],
        ['','_kp','_ki','_kd','_dacmin','_setpoint']
    ) + striter_combine(
        ['elTWC','el083','el021','elCMP'],
        ['_r100','_dtdr','_l','_w','_d','_shape','_f']
    ) + striter_combine(['elTWC','el083','el021'], ['_s','_o']) +
    striter_combine(['caldate','calduedate'], ['_month','_day','_year'])
)
cprb_longnames = (
    ['id','serial number','chip id','revision'] +
    striter_combine(
        ['element TWC', 'element 083', 'element 021', 'element CMP',
         'element DCE'],
        ['',' integral control loop gain', ' differential control loop gain',
         ' proportional control loop gain', ' dac minimum',
         ' setpoint temperature']
    ) +
    striter_combine(
        ['element TWC','element 083','element 021','element CMP'],
        [' r100 calibration parameter', ' dtdr calibration parameter',
         ' length',' width',' depth', ' shape',' fixture resistance']
    ) +
    striter_combine(
        ['element TWC','element 083','element 021'],
        [' slope correction (K1)',' offset correction (K2)']
    ) +
    striter_combine(
        ['calibration','calibration due'], [' month',' day',' year']
    )
)
cprb_units = (
    [''] * 4 + ['', '', '', '', '', 'deg C'] * 5 +
    ['milliohm', 'deg C/milliohm', 'mm', 'mm', 'mm', '', 'milliohm'] * 4 +
    ['', ''] * 3 + ['', '', ''] * 2
)

# Define structure fo identifier cmcb
cmcb_types = ['S4', 'S5', 'S16', 'S4'] + ['int'] * 4
cmcb_names = ['id', 'sn', 'chipid', 'rev', 'cablelen', 'ele3res', 'cmpres',
              'dceres']
cmcb_longnames = ['id', 'serial number', 'chip id', 'revision',
                   'cable length', 'element 3 resistance',
                   'compensation resistance', 'deice resistance']
cmcb_units = [''] * 4 + ['ft', 'milliohm', 'milliohm', 'milliohm']

# Define structure fo identifier cmcb
cscb_types = ['S4', 'S5', 'S16', 'S4'] + ['int'] * 3
cscb_names = ['id', 'sn', 'chipid', 'rev', 'cablelen', 'ele1res', 'ele2res']
cscb_longnames = ['id', 'serial number', 'chip id', '', 'cable length',
                  'element 1 resistance', 'element 2 resistance']
cscb_units = [''] * 4 + [ 'ft', 'milliohm', 'milliohm']
