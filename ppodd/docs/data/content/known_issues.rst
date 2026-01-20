============
Known Issues
============

The following issues are known to exist, or have existed, in the FAAM core data product.

---------------------------
Airspeed calculation errors
---------------------------

Prior to software version 25.1.0, and dating back many years, there was an error in the
calculation of Mach number from the aircraft's air data system. The error
was the use of the equation

.. math::

    \text{IAS} = a\cdot M \cdot\sqrt{\frac{P}{P_0}}

where :math:`a` is the speed of sound at sea level, :math:`M` is the Mach number,
:math:`P` is the static pressure, and :math:`P_0` is the standard sea level pressure.
However this equation defines equivalent airspeed (EAS), not IAS.

This error led to a systematic error in the calculation of true airspeed of up to
around 2.5\% at the operating ceiling of the aircraft, and an error in dynamic pressure
of up to around 7\%. 

Other derived quantities that depend on true airspeed or dynamic pressure, such as
wind components and true air temperatures, were also affected.

Further details are available in the FAAM documents

- FAAM-000360-RPT: ARA Airspeed Investigation
- FAAM-000380-RPT: Airspeed Processing Errors
 
which are available from FAAM on request.
