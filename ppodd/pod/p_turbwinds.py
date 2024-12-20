import logging
from functools import reduce

import numpy as np

from .base import PPBase, register_pp
from .shortcuts import _c, _o, _r, _z
from ..decades import DecadesVariable, DecadesBitmaskFlag, flags
from ..utils import slrs, get_range_flag

TAS_VALID_RANGE = (50, 250)
AOA_VALID_RANGE = (-10, 15)
AOSS_VALID_RANGE = (-5, 5)
MACH_VALID_RANGE = (0.05, 0.8)
WIND_VALID_RANGE = (-60, 60)

logger = logging.getLogger(__name__)

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


@register_pp('core')
class TurbulenceProbe(PPBase):
    r"""
    This module produces flow angles and airspeed from the turbulence probe 
    pressure differentials, and combines these with inertial/GPS measurements
    to derive 3-dimensional winds. These two processes are performed iteratively
    so that covariances between inferred winds and inertial measurements can be
    used to calculate on-the-fly flow angle correction terms. Alternatively,
    flow angle corrections can be provided in the flight constants.

    **Flow angles, dynamic pressure, and TAS**

    Initial estimates for angle of attack, :math:`\alpha`, and angle of sideslip, :math:`\beta`,
    are given by

    .. math::
        \alpha &= \left(P_a / q - a_0\right) / a_1,\\
        \beta  &= \left(P_b / q - b_0\right) / b_1.

    Here, :math:`P_a` and :math:`P_b` are the pressure differentials between the 
    top/bottom and left/right probe ports, :math:`a_n` and :math:`b_n` are 
    calibration coefficients derived from flight data, and :math:`q` is the 
    RVSM-derived dynamic pressure.

    Correction terms to these flow angles, required as stagnation points do not
    correspond exactly with probe ports, calculated by BAeS during the probe 
    commissioning, are given by

    .. math::
        \delta_\alpha &= 0.0273 + \alpha\left(-0.0141 + \alpha\left(0.00193 - 0.000052\alpha\right)\right),\\
        \delta_\beta &= 0.00076172\beta^2.

    An initial estimate of dynamic pressure from the turbulence probe, :math:`q_t`, is then given by

    .. math::
        q_t = \Delta_{P_0S_{10}} + q(\delta_\alpha + \delta_\beta),

    where :math:`\Delta_{P_0S_{10}}` is the pressure differential between :math:`P_0` and :math:`S_{10}`.

    This whole process is then iteratively repeated with :math:`q = q_t`, either until the difference in 
    both flow angles between iterations is less than 0.2 degrees, or 5 iterations have been completed.

    Mach number, :math:`M`, is given by 

    .. math::
        M = \sqrt{5\left(\left(1 + \frac{q_t}{p}\right)^\frac{2}{7} - 1\right)},

    where :math:`p` is the static pressure from the RVSM system.

    True airspeed, TAS, is then given by

    .. math::
        \text{TAS} = \text{TAS}_c C_0 M \sqrt{T_\text{air} / T_0},

    where :math:`\text{TAS}_c` is a correction term derived from flight data, :math:`C_0` is the speed of
    sound at ICAO STP, :math:`T_\text{air}` is the (deiced) true air temperature, and :math:`T_0` is the
    temperature at ICAO STP.

    **Wind vector**

    First let us define rotation matricies which provide the transformations between geographical
    and aircraft-relative coordinate systems.

    .. math::
        \mathbf{A_1} &= \begin{bmatrix}
                1 & 0       & 0 \\
                0 & \cos(r) & -\sin(r) \\
                0 & sin(r)  & cos(r)
            \end{bmatrix},\\ 
        \mathbf{A_2} &= \begin{bmatrix}
                \cos(\theta) & 0 & -\sin(\theta) \\
                0            & 1 & 0 \\
                \sin(\theta) & 0 & \cos(\theta)
            \end{bmatrix}, \\
        \mathbf{A_3} &= \begin{bmatrix}
                \cos(\phi)  & \sin(\phi) & 0 \\
                -\sin(\phi) & \cos(\phi) & 0 \\
                0           & 0          & 1
            \end{bmatrix},

    where :math:`r`, :math:`\theta`, and :math:`\phi` are the aircraft roll, pitch, and yaw, respectively. We
    can combine these into a single transformation tensor, :math:`\mathbf{R} = (\mathbf{A_3}\cdot\mathbf{A_2})
    \cdot\mathbf{A_1}`.

    The wind vector :math:`\mathbf{U}`, is given by 

    .. math::
        \mathbf{U} = \mathbf{V} + \mathbf{T} + \mathbf{L},

    where :math:`\mathbf{V}` is the aircraft velocity vector, :math:`\mathbf{T}` is the aircraft TAS
    vector, relative to the earth, and :math:`\mathbf{L}` is a correction term which accounts for the
    offset between the aircraft inertial measurement and the turbulence probe.

    The correction term :math:`\mathbf{L}` is given by

    .. math::
        \mathbf{L} = \mathbf{L_L} \times \mathbf{L_R},

    where :math:`\mathbf{L_R} = \mathbf{R} \cdot \mathbf{l}` and :math:`\mathbf{L_L} = \mathbf{L_1} +
    \mathbf{L_2} + \mathbf{L_3}`.

    Here :math:`\mathbf{l}` is the position vector describing the offset of the inertial measurement from
    the turbulence probe, and

    .. math::
        \mathbf{L_1} &= [0, 0, -\dot\phi],\\
        \mathbf{L_2} &= \mathbf{A_3} \cdot [0, \dot\theta, 0],\\
        \mathbf{L_3} &= (\mathbf{A_3}\cdot\mathbf{A_2})\cdot[\dot{r}, 0, 0],

    where an overdot represents a time derivative.

    The TAS term :math:`\mathbf{T}` is given by

    .. math::
        R = \mathbf{R} \cdot [-T_p, -T_p\tan(\beta), T_p\tan(\alpha)],

    where

    .. math::
        T_p = \frac{\text{TAS}}{\sqrt{1 + \tan(\beta)^2 + \tan(\alpha)^2}}.

    **Flow angle corrections**

    The derivation of the variables above assumes no flow angle corrections, which are generally required.
    We take the approach of calculating flow angle corrections on line. This is done in an iterative loop
    where, after calculation of the flow angles and wind vector, flow angle corrections are calulated by
    1) minimising the covariance between platform roll and vertical wind in the turns, and 2) minimising the
    mean vertical wind when the platform is straight and level. Flow angles and are then recalculated, and
    so on until the flow angle corrections converge suitably. This has been seen to occur rapidly, and only
    two iterations after the initial calculations are performed.    
    """

    # TODO: Is it OK this module replace p_turb and p_winds for old flights, or
    # should we implement VALID_AFTER (1/10/2020)??

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

    @staticmethod
    def test():
        n = 900
        n3 = 300
        return {
            'AOA_A0': ('const', [.25, .5, -.75]),
            'AOA_A1': ('const', [-.05, -.1, .1]),
            'AOSS_B0': ('const', [0]),
            'AOSS_B1': ('const', [.05]),
            'TOLER': ('const', 0.05),
            'TASCOR1': ('const', 1),
            'ALPHA_COR': ('const', [0, 1]),
            'BETA_COR': ('const', [0, 1]),
            'INSPOSN': ('const', [16, -1, -.5]),
            'IAS_RVSM': ('data', 100 * _o(n), 32),
            'TAT_DI_R': ('data', 290 * _o(n), 32),
            'PS_RVSM': ('data', 800 * _o(n), 32),
            'Q_RVSM': ('data', 100 * _o(n), 32),
            'P0_S10': ('data', 100 * _o(n), 32),
            'PA_TURB': ('data', .5 * _o(n), 32),
            'PB_TURB': ('data', .01 * _o(n), 32),
            'VELN_GIN': ('data', 100 * _o(n) + _r(n), 32),
            'VELE_GIN': ('data', _o(n), 32),
            'VELD_GIN': ('data', _z(n), 32),
            'ROLL_GIN': ('data', _c([_z(n3), 15*_o(n3), _z(n3)]), 32),
            'PTCH_GIN': ('data', 5 * _o(n), 32),
            'HDG_GIN': ('data', _z(n), 32),
            'ROLR_GIN': ('data', _z(n), 32),
            'PITR_GIN': ('data', _z(n), 32),
            'HDGR_GIN': ('data', _z(n), 32),
            'WOW_IND': ('data', _z(n), 32)
        }

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
        d['WOW_IND'] = d['WOW_IND'].ffill()

        consts = {
            i: self.dataset[i] for i in
            ('AOA_A0', 'AOA_A1', 'AOSS_B0', 'AOSS_B1', 'TASCOR1', 'TOLER',
             'INSPOSN', 'BETA_COR', 'ALPHA_COR')
        }

        calc_ab_cor = self.dataset.constants.get('TP_CALC_AB_COR', False)

        if calc_ab_cor:
            try:
                _slrs = reduce(
                    lambda x, y: x.union(y),
                    slrs(d.WOW_IND, d.PS_RVSM, d.ROLL_GIN, freq=32)
                )
            except TypeError:
                _slrs = None

            _in_roll = d.ROLL_GIN.abs() > 10

        for i in range(3):
            if not calc_ab_cor:
                # Just use a, b corrections specified in constants
                continue

            if self.test_mode:
                continue

            if _slrs is None:
                logger.warning('No SLRs identified. Cannot estimate alpha '
                               'correction term')

            ws = []
            alphas = [-2, 2]

            if _slrs is not None:
                for alpha in alphas:
                    consts['ALPHA_COR'] = [alpha, 1]
                    d = p_turb(d.copy(deep=True), consts)
                    d = p_winds(d.copy(deep=True), consts)
                    ws.append(d.W_C[_slrs].mean())
                fit = np.polyfit(ws, alphas, 1)
                consts['ALPHA_COR'] = [np.polyval(fit, 0), 1]

            covs = []
            betas = [-2, 2]

            roll_index = (
                np.isfinite(d.ROLL_GIN[_in_roll]) &
                np.isfinite(d.W_C[_in_roll])
            )

            for beta in betas:
                consts['BETA_COR'] = [beta, 1]

                d = p_turb(d.copy(deep=True), consts)
                d = p_winds(d.copy(deep=True), consts)

                covs.append(
                    (np.cov(
                        d.W_C[_in_roll][roll_index],
                        d.ROLL_GIN[_in_roll][roll_index]
                    ))[0, 1]
                )

            try:
                fit = np.polyfit(covs, betas, 1)
            except Exception:
                if not self.test_mode:
                    raise
                fit = [0, 1]
            consts['BETA_COR'] = [np.polyval(fit, 0), 1]

            logger.info('alpha, beta corrections = {}, {}'.format(
                consts['ALPHA_COR'][0], consts['BETA_COR'][0]
            ))

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
