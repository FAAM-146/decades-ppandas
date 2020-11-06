from functools import reduce

import numpy as np

from .base import PPBase
from ..decades import DecadesVariable, DecadesBitmaskFlag, flags
from ..utils import slrs, get_range_flag

TAS_VALID_RANGE = (50, 250)
AOA_VALID_RANGE = (-10, 15)
AOSS_VALID_RANGE = (-5, 5)
MACH_VALID_RANGE = (0.05, 0.8)
WIND_VALID_RANGE = (-60, 60)

def mach(p, q):
    return np.sqrt(5*((1 + q / p)**(2./7.) - 1))

def _rotation_arrays(r, th, ph):
    """
    Return matricies which define transformations between aircraft and
    geographical frames of reference.

    A1 = | 1     0       0       |
         | 0     cos(r)  -sin(r) |
         | 0     sin(r)  cos(r)  |

    A2 = | cos(th)  0   -sin(th) |
         | 0        1   0        |
         | sin(th)  0   cos(th)  |

    A3 = | cos(ph)  sin(ph) 0 |
         | -sin(ph) cos(ph) 0 |
         | 0        0       1 |

    Args:
        r: the aircraft roll, in radians
        th: the aircraft pitch, in radians
        ph: the aircraft heading, in radians

    Returns:
        A1, A2, A3: Roll, pitch and heading correction tensors.
    """
    n = len(r)

    A1 = np.array([
        [np.ones(n), np.zeros(n), np.zeros(n)],
        [np.zeros(n), np.cos(r), -np.sin(r)],
        [np.zeros(n), np.sin(r), np.cos(r)]
    ])

    A2 = np.array([
        [np.cos(th), np.zeros(n), -np.sin(th)],
        [np.zeros(n), np.ones(n), np.zeros(n)],
        [np.sin(th), np.zeros(n), np.cos(th)]
    ])

    A3 = np.array([
        [np.cos(ph), np.sin(ph), np.zeros(n)],
        [-np.sin(ph), np.cos(ph), np.zeros(n)],
        [np.zeros(n), np.zeros(n), np.ones(n)]
    ])

    return A1, A2, A3


def p_turb(d, consts):
    _mach = mach(d['PS_RVSM'], d['Q_RVSM'])

    a0 = np.polyval(consts['AOA_A0'][::-1], _mach)
    a1 = np.polyval(consts['AOA_A1'][::-1], _mach)
    b0 = np.polyval(consts['AOSS_B0'][::-1], _mach)
    b1 = np.polyval(consts['AOSS_B1'][::-1], _mach)

    aoa = (d['PA_TURB'] / d['Q_RVSM'] - a0) / a1
    aoss = (d['PB_TURB'] / d['Q_RVSM'] - b0) / b1

    aoa[d['IAS_RVSM'] < 50] = np.nan
    aoss[d['IAS_RVSM'] < 50] = np.nan

    dcpa = 0.0273 + aoa * (-0.0141 + aoa * (0.00193 - aoa * 5.2E-5))
    dcpb = aoss * (aoss * 7.6172E-4)

    q = d['P0_S10'] + (dcpa + dcpb) * d['Q_RVSM']

    amach = mach(d['PS_RVSM'], q)

    itern = 0
    while True:
        itern += 1

        a0 = np.polyval(consts['AOA_A0'][::-1], amach)
        a1 = np.polyval(consts['AOA_A1'][::-1], amach)
        b0 = np.polyval(consts['AOSS_B0'][::-1], amach)
        b1 = np.polyval(consts['AOSS_B1'][::-1], amach)

        aoa_new  = (d['PA_TURB'] / q - a0) / a1
        aoss_new = (d['PB_TURB'] / q - b0) / b1

        d_aoa = aoa_new - aoa
        d_aoss = aoss_new - aoss

        aoa_new[d['IAS_RVSM'] < 50] = np.nan
        aoss_new[d['IAS_RVSM'] < 50] = np.nan

        aoa = aoa_new
        aoss = aoss_new

        dcpa = 0.0273 + aoa * (-0.0141 + aoa * (0.00193- aoa * 5.2E-5))
        dcpb = aoss * (aoss * 7.6172E-4)

        q = d['P0_S10'] + (dcpa + dcpb) * q

        amach = mach(d['PS_RVSM'], q)

        tol = 0.2
        if np.max(np.abs(d_aoa)) < tol and np.max(np.abs(d_aoss)) < tol:
            break

        if itern > 5:
            break

    tas = (
        consts['TASCOR1'] * 340.294 * amach
        * np.sqrt(d['TAT_DI_R'] / 288.15)
    )

    aoss = aoss * consts['BETA_COR'][1] + consts['BETA_COR'][0]
    aoa = aoa * consts['ALPHA_COR'][1] + consts['ALPHA_COR'][0]

    d['AOA'] = aoa
    d['AOSS'] = aoss
    d['TAS'] = tas
    d['PSP_TURB'] = q

    return d

def p_winds(d, consts):
    l = consts['INSPOSN']
    n = len(d)

    MATMUL = 'ij...,jk...->ik...'
    DOT = 'ij...,j...->i...'

     # Create compactly named input variable, with required reansformations.
    vn = d['VELE_GIN']
    ve = d['VELN_GIN']
    vz = -d['VELD_GIN']
    r = np.deg2rad(d['ROLL_GIN'])
    th = np.deg2rad(d['PTCH_GIN'])
    ph = np.deg2rad(d['HDG_GIN']-90)
    rdot = np.deg2rad(d['ROLR_GIN'])
    thdot = np.deg2rad(d['PITR_GIN'])
    phdot = np.deg2rad(d['HDGR_GIN'])
    tas = d['TAS']
    alpha = np.deg2rad(d['AOA'])
    beta = np.deg2rad(d['AOSS'])

    # Get the transformation matricies required to rotate from geographical
    # to platform relative coordinate systems
    A1, A2, A3 = _rotation_arrays(r, th, ph)

    # Aircraft velocity vector in, geo relative
    V = np.array([vn, ve, vz])

    # Build compound rotation matrix to convert aircraft <-> geo coords.
    # This is (A3.A2).A1
    R = np.einsum(MATMUL, np.einsum(MATMUL, A3, A2), A1)

    # Transform the aircraft TAS vector to geo coords
    tas_p = tas / (1 + np.tan(beta)**2 + np.tan(alpha)**2)**.5
    TAS = np.array([-tas_p, -tas_p * np.tan(beta), tas_p * np.tan(alpha)])
    TASR = np.einsum(DOT, R, TAS)

    # Corrections for the offset between the gust probe (radome) and the
    # GIN
    LEV_L1 = np.array([np.zeros(n), np.zeros(n), -phdot])
    LEV_L2 = np.einsum(
        DOT, A3, np.array([np.zeros(n), -thdot, np.zeros(n)])
    )
    LEV_L3 = np.einsum(
        DOT, np.einsum(MATMUL, A3, A2),
        np.array([rdot, np.zeros(n), np.zeros(n)])
    )

    LEV_L = LEV_L1 + LEV_L2 + LEV_L3
    LEV_R = np.einsum(DOT, R, l)
    LEV = np.cross(LEV_L.T, LEV_R.T).T

    # Wind velocity vector is the sum of the aircraft ground velocity
    # vector, the geo relative TAS vector and the GIN offset correction.
    U = V + TASR + LEV

    d['U_C'] = U[0, :]
    d['V_C'] = U[1, :]
    d['W_C'] = U[2, :]

    return d


class TurbulentWinds(PPBase):

    inputs = [
        'AOA_A0',
        'AOA_A1',
        'AOSS_B0',
        'AOSS_B1',
        'ALPHA_COR',
        'BETA_COR',
        'TOLER',
        'TASCOR1',
        'INSPOSN',
        'IAS_RVSM',
        'TAT_DI_R',
        'PS_RVSM',
        'Q_RVSM',
        'P0_S10',
        'PA_TURB',
        'PB_TURB',
        'VELN_GIN',
        'VELE_GIN',
        'VELD_GIN',
        'ROLL_GIN',
        'PTCH_GIN',
        'HDG_GIN',
        'ROLR_GIN',
        'PITR_GIN',
        'HDGR_GIN',
        'WOW_IND'
    ]

    def declare_outputs(self):
        self.declare(
            'AOA',
            units='degree',
            frequency=32,
            long_name=('Angle of attack from the turbulence probe (positive, '
                       'flow upwards wrt a/c axes)')
        )

        self.declare(
            'AOSS',
            units='degree',
            frequency=32,
            long_name=('Angle of sideslip from the turbulence probe '
                       '(positive, flow from left)')
        )

        self.declare(
            'TAS',
            units='m s-1',
            frequency=32,
            long_name='True airspeed (dry-air) from turbulence probe',
            standard_name='platform_speed_wrt_air'
        )

        self.declare(
            'PSP_TURB',
            units='hPa',
            frequency=32,
            long_name=('Pitot-static pressure from centre-port measurements '
                       'corrrected for AoA and AoSS')
        )

        self.declare(
            'V_C',
            units='m s-1',
            frequency=32,
            long_name='Northward wind component from turbulence probe and GIN',
            standard_name='northward_wind'
        )

        self.declare(
             'U_C',
             units='m s-1',
             frequency=32,
             long_name='Eastward wind component from turbulence probe and GIN',
             standard_name='eastward_wind'
         )

        self.declare(
            'W_C',
            units='m s-1',
            frequency=32,
            long_name='Vertical wind component from turbulence probe and GIN',
            standard_name='upward_air_velocity'
        )

    def process(self):

        self.get_dataframe(
            method='onto', index=self.dataset['PS_RVSM'].index,
            circular=('HDG_GIN',)
        )
        d = self.d
        d['WOW_IND'].fillna(method='ffill', inplace=True)

        consts = {
            i: self.dataset[i] for i in
            ('AOA_A0', 'AOA_A1', 'AOSS_B0', 'AOSS_B1', 'TASCOR1', 'TOLER',
             'INSPOSN', 'BETA_COR', 'ALPHA_COR')
        }

        _slrs = reduce(
            lambda x, y: x.union(y),
            slrs(d.WOW_IND, d.PS_RVSM, d.ROLL_GIN, freq=32)
        )

        _in_roll = d.ROLL_GIN.abs() > 10

        for i in range(2):
            ws = []
            alphas = [-2, 2]

            for alpha in alphas:
                consts['ALPHA_COR'] = [alpha, 1]
                d = p_turb(d.copy(deep=True), consts)
                d = p_winds(d.copy(deep=True), consts)
                ws.append(d.W_C.loc[_slrs].mean())
            fit = np.polyfit(ws, alphas, 1)
            consts['ALPHA_COR'] = [np.polyval(fit, 0), 1]

            covs = []
            betas = [-2, 2]

            for beta in betas:
                consts['BETA_COR'] = [beta, 1]

                d = p_turb(d.copy(deep=True), consts)
                d = p_winds(d.copy(deep=True), consts)
                covs.append((np.cov(d.W_C[_in_roll], d.ROLL_GIN[_in_roll]))[0, 1])


            fit = np.polyfit(covs, betas, 1)
            consts['BETA_COR'] = [np.polyval(fit, 0), 1]

            print(consts['ALPHA_COR'][0], consts['BETA_COR'][0])

        d = p_turb(d, consts)
        d = p_winds(d, consts)

        # Create outputs
        u_out = DecadesVariable(
            d['U_C'], name='U_C', flag=DecadesBitmaskFlag
        )
        w_out = DecadesVariable(
            d['W_C'], name='W_C', flag=DecadesBitmaskFlag
        )
        v_out = DecadesVariable(
            d['V_C'], name='V_C', flag=DecadesBitmaskFlag
        )
        tas_out = DecadesVariable(
            d['TAS'], name='TAS', flag=DecadesBitmaskFlag
        )
        aoa_out = DecadesVariable(
            d['AOA'], name='AOA', flag=DecadesBitmaskFlag
        )
        aoss_out = DecadesVariable(
            d['AOSS'], name='AOSS', flag=DecadesBitmaskFlag
        )
        psp_turb_out = DecadesVariable(
            d['PSP_TURB'], name='PSP_TURB', flag=DecadesBitmaskFlag
        )

        tas_flag = get_range_flag(d.TAS, TAS_VALID_RANGE)
        u_flag = get_range_flag(d.U_C, WIND_VALID_RANGE)
        v_flag = get_range_flag(d.V_C, WIND_VALID_RANGE)
        aoa_flag = get_range_flag(d.AOA, AOA_VALID_RANGE)
        aoss_flag = get_range_flag(d.AOSS, AOSS_VALID_RANGE)
        mach_flag = get_range_flag(mach(d.PS_RVSM, d.PSP_TURB),
                                   MACH_VALID_RANGE)

        _out_range_desc = (
            '{} is outside the specified valid range [({}, {})]{}'
        )
        tas_desc = _out_range_desc.format('TAS', *TAS_VALID_RANGE, ' m/s')
        aoa_desc = _out_range_desc.format('AOA', *AOA_VALID_RANGE, ' deg')
        aoss_desc = _out_range_desc.format('AOSS', *AOSS_VALID_RANGE, ' deg')
        mach_desc = _out_range_desc.format('Mach #', *MACH_VALID_RANGE, '')
        u_desc = _out_range_desc.format('U_C', *WIND_VALID_RANGE, ' m/s')
        v_desc = _out_range_desc.format('V_C', *WIND_VALID_RANGE, ' m/s')
        mach_meaning = 'mach out of range'

        tas_out.flag.add_mask(tas_flag, flags.OUT_RANGE, tas_desc)
        aoa_out.flag.add_mask(aoa_flag, flags.OUT_RANGE, aoa_desc)
        aoss_out.flag.add_mask(aoss_flag, flags.OUT_RANGE, aoss_desc)
        u_out.flag.add_mask(u_flag, flags.OUT_RANGE, u_desc)
        v_out.flag.add_mask(v_flag, flags.OUT_RANGE, v_desc)

        for var in (aoa_out, aoss_out, psp_turb_out, u_out, v_out, w_out):
            var.flag.add_mask(mach_flag, mach_meaning, mach_desc)

        for var in (tas_out, aoa_out, aoss_out, psp_turb_out, u_out, v_out,
                    w_out):
            var.flag.add_mask(d.WOW_IND, flags.WOW, 'aircraft on ground')

            self.add_output(var)
