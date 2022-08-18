import numpy as np
import datetime

from ..decades import DecadesVariable
from ..decades.flags import DecadesBitmaskFlag
from .base import PPBase
from .shortcuts import _o, _z

# TODO: Should we reregister this for docs, or not?
class TurbWinds(PPBase):

    DEPRECIATED_AFTER = datetime.date(2020, 10, 1)

    inputs = [
        'INSPOSN',
        'TAS',
        'VELN_GIN',
        'VELE_GIN',
        'VELD_GIN',
        'ROLL_GIN',
        'PTCH_GIN',
        'HDG_GIN',
        'ROLR_GIN',
        'PITR_GIN',
        'HDGR_GIN',
        'AOA',
        'AOSS'
    ]

    @staticmethod
    def test():
        return {
            'INSPOSN': ('const', [16, -.8, -.4]),
            'TAS': ('data', 120 * _o(100)),
            'VELN_GIN': ('data', 100 * _o(100)),
            'VELE_GIN': ('data', 100 * _o(100)),
            'VELD_GIN': ('data', _z(100)),
            'ROLL_GIN': ('data', _z(100)),
            'PTCH_GIN': ('data', 6 * _o(100)),
            'HDG_GIN': ('data', _z(100)),
            'ROLR_GIN': ('data', _z(100)),
            'PITR_GIN': ('data', _z(100)),
            'HDGR_GIN': ('data', _z(100)),
            'AOA': ('data', 6 * _o(100)),
            'AOSS': ('data', _z(100))
        }

    def declare_outputs(self):
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

    def _rotation_arrays(self, r, th, ph):
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

    def process(self):
        self.get_dataframe()
        d = self.d

        l = np.array(self.dataset['INSPOSN'])
        n = len(d)

        # Define Einstein notation for required matrix operations
        MATMUL = 'ij...,jk...->ik...'
        DOT = 'ij...,j...->i...'

        # Create compactly named input variable, with required reansformations.
        vn = d.VELE_GIN
        ve = d.VELN_GIN
        vz = -d.VELD_GIN
        r = np.deg2rad(d.ROLL_GIN)
        th = np.deg2rad(d.PTCH_GIN)
        ph = np.deg2rad(d.HDG_GIN-90)
        rdot = np.deg2rad(d.ROLR_GIN)
        thdot = np.deg2rad(d.PITR_GIN)
        phdot = np.deg2rad(d.HDGR_GIN)
        tas = d.TAS
        alpha = np.deg2rad(d.AOA)
        beta = np.deg2rad(d.AOSS)

        # Get the transformation matricies required to rotate from geographical
        # to platform relative coordinate systems
        A1, A2, A3 = self._rotation_arrays(r, th, ph)

        # Aircraft velocity vector in, geo relative
        V = np.array([vn, ve, vz])

        # Build compound rotation matrix to convert aircraft <-> geo coords. 
        # This is (A3.A2).A1
        R = np.einsum(MATMUL, np.einsum(MATMUL, A3, A2), A1)

        # Transform the aircraft TAS vector to geo coords
        tas_p = tas / (1 + np.tan(beta)**2 + np.tan(alpha**2))**.5
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

        u = DecadesVariable(d.U_C, flag=DecadesBitmaskFlag)
        v = DecadesVariable(d.V_C, flag=DecadesBitmaskFlag)
        w = DecadesVariable(d.W_C, flag=DecadesBitmaskFlag)

        tas_flag = self.dataset['TAS'].flag().reindex(d.index) > 0
        tas_flag.fillna(1, inplace=True)

        ao_flag = (
            (self.dataset['AOSS'].flag().reindex(d.index) > 0) |
            (self.dataset['AOA'].flag().reindex(d.index) > 0)
        )
        ao_flag.fillna(1, inplace=True)

        # Apply basic flagging
        for var in (u, v, w):
            var.flag.add_mask(
                tas_flag, 'TAS FLAGGED',
                description='True airspeed has non-zero flag'
            )

            var.flag.add_mask(
                ao_flag, 'AOA_AOSS_FLAGGED',
                description=('Angle of attack and/or angle of sideslip has '
                             'a non-zero flag')
            )

        for var in (u, v, w):
            self.add_output(var)
