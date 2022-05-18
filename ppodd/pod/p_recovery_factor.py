"""
Provides a processing module which calculates a mach number for moist
air.
"""
# pylint: disable=invalid-name
import numpy as np

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..utils.calcs import sp_mach
from .base import PPBase, register_pp
from .shortcuts import _o


@register_pp('core')
class RecoveryFactor(PPBase):
    r"""
    This module produces a Mach dependent recovery factor for the deiced and
    non-deiced Rosemount housings, which house the temperature probes.

    The recovery factor for the non-deiced housing is given by Rosemount
    Aerospace as

    .. math::

        \eta_{nd} = a_1 M + a_2 M^2

    where :math:`a_1 = 0.0014054157` and :math:`a_2 = -0.00060943146`

    The recovery factor for the deiced housing is calculated from in-flight
    data, under the assumption that the recovery factor given by Rosemount is
    correct. It is given by

    .. math::

        \eta_{di} = 1 - \alpha\left(1 - \eta_{nd}\right),

    where :math:`\alpha = 0.9989` is the average ratio of deiced and non-deiced
    indicated temperatures. Further information is available in the FAAM
    Recovery Factor report (H. Price, document # FAAM013006).

    Uncertainties are given by

    .. math::

        \alpha_U &= 0.0006\\
        \eta_{nd,U} &= -0.0003458119 M^2 + 0.00059345748 M\\
        \eta_{di,U} &= ((\eta_{nd}-1)^2\alpha_U^2 + \alpha^2
        \eta_{nd,U}^2)^\frac{1}{2}

    """

    inputs = [
        'PS_RVSM',
        'Q_RVSM',
        'CORCON_di_temp',
        'CORCON_ndi_temp'
    ]

    @staticmethod
    def test():
        """
        Provide dummy input data for testing.
        """
        n = 100
        return {
            'PS_RVSM': ('data', 850 * _o(n), 32),
            'Q_RVSM': ('data', 75 * _o(n), 32),
            'CORCON_di_temp': ('data', _o(n), 32),
            'CORCON_ndi_temp': ('data', _o(n), 32)
        }

    def declare_outputs(self):
        """
        Declare outputs created by this module.
        """
        self.declare(
            'ETA_ND',
            units='1',
            frequency=32,
            long_name=('Variable recovery factor for non-deiced Rosemount '
                       'housing'),
            write=False
        )

        self.declare(
            'ETA_DI',
            units='1',
            frequency=32,
            long_name=('Variable recovery factor for deiced Rosemount '
                       'housing'),
            write=False
        )

        self.declare(
            'ETA_ND_CU',
            units='1',
            frequency=32,
            long_name=('Uncertainty in recovery factor for deiced '
                       'Rosemount housing'),
            write=False,
            coverage_content_type='auxiliaryInformation',
            flag=None
        )

        self.declare(
            'ETA_DI_CU',
            units='1',
            frequency=32,
            long_name=('Uncertainty in recovery factor for non-deiced '
                       'Rosemount housing'),
            coverage_content_type='auxiliaryInformation',
            write=False,
            flag=None
        )

    def process(self):
        """
        Processing entry hook.
        """
        self.get_dataframe()
        d = self.d

        mach = sp_mach(d.Q_RVSM, d.PS_RVSM)
        eta_nd = -0.00060943146 * mach**2 +0.0014054157 * mach
        eta_nd = eta_nd.reindex(d.CORCON_ndi_temp.index).fillna(0)

        alpha = 0.9989
        eta_di = 1.0 - alpha * (1.0 - eta_nd)
        eta_di = eta_di.reindex(d.CORCON_di_temp.index).fillna(0)

        # Uncertainties
        u_alpha = 0.0006
        u_eta_nd = -0.0003458119 * mach**2. + 0.00059345748 * mach
        u_eta_di = (
            (eta_nd - 1.)**2.0 * u_alpha**2. + alpha**2. * u_eta_nd**2.
        )**0.5

        # Add outputs
        self.add_output(DecadesVariable(eta_nd, name='ETA_ND', flag=None))
        self.add_output(DecadesVariable(eta_di, name='ETA_DI', flag=None))
        self.add_output(DecadesVariable(u_eta_nd, name='ETA_ND_CU', flag=None))
        self.add_output(DecadesVariable(u_eta_di, name='ETA_DI_CU', flag=None))
