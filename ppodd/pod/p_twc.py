import logging

import numpy as np
import pandas as pd

from ppodd.decades.flags import DecadesClassicFlag
from ppodd.decades.variable import DecadesVariable

from .base import PPBase, register_pp


logger = logging.getLogger(__name__)


F = 0.93
Kv = 427.0
p0 = 1013.25
t0 = 273.15
u0 = 0.2095

def thermistor(data, A, B, C, D):
    return B / np.log(C - data) - np.log(data + D) + A


def pd_polyval(p, x):
    index = x.index
    return pd.Series(np.polyval(p, x), index=index)

def get_KO(p):
    return 0.304 + 0.351 * p * F / p0

def make_fit(p, t, tsam, tdet, vmr):
    
    KO = get_KO(p)

    svp  = saturation_vapour_pressure(t)
    vp1 = vmr2vp(vmr / 1e6, p)

    mask = ((vp1 / svp) < 0.7) & ~np.isnan(vp1) & ~np.isnan(t) & ~np.isnan(tsam)

    fitted = (vp1 / tsam) + (KO * u0 + p / (Kv * tsam))

    try:
        fit = np.polyfit(
            tdet[mask],
            fitted[mask],
            # (vp1[mask] / tsam[mask]) + (KO[mask] * u0 + p[mask] / (Kv * tsam[mask])),
            1
        )
    except Exception as e:
        fit = 0

    import matplotlib.pyplot as plt 
    plt.plot(tdet, fitted, 'k+')
    plt.plot(tdet[mask], fitted[mask], '.')
    plt.plot(tdet, np.polyval(fit, tdet))
    plt.show()

    return fit
    
def vmr2vp(vmr, p):
    import matplotlib.pyplot as plt
    vp = p * (vmr / (1.0 + vmr))
    plt.plot(vp)
    plt.show()
    return vp

def vp2vmr(vp,p):    
    vmr=vp/(p-vp)
    return vmr

def vmr_mmr(vmr):
    MMR  =  622.0 * vmr 
    return MMR

def vp2dp(vp,p=None,temp=None,enhance=False):
    """
    Convert a volume mixing ratio to a dew point ( and vapour pressure )
    Using ITS-90 correction of Wexler's formula
    Optional enhancement factors for non ideal 
    """  
    vp=np.atleast_1d(vp)
    c=np.array([-9.2288067e-06, 0.46778925, -20.156028, 207.98233],dtype='f8')
    d=np.array([-7.5172865e-05, 0.0056577518, -0.13319669, 1],dtype='f8')
    lnes=np.log(vp*1e2)
    dp=np.polyval(c,lnes)/np.polyval(d,lnes)
    if(enhance and len(p)>0) :
        if(len(temp)==0):
            temp=dp
        A=np.array([8.5813609e-09, -6.7703064e-06, 0.001807157, -0.16302041],dtype='f8')
        B=np.array([6.3405286e-07, -0.00077326396, 0.34378043, -59.890467],dtype='f8')
        alpha=np.polyval(A,temp)
        beta=np.exp(np.polyval(B,temp))
        ef=np.exp(alpha*(1-vp/p)+beta*(p/vp-1))
        vp=vp/ef
        lnes=np.log(vp*1e2)
        dp=np.polyval(c,lnes)/np.polyval(d,lnes)
    return dp

def saturation_vapour_pressure(dp):#,p=[],temp=[],enhance=False):
    """
    Convert a dew point to a volume mixing ratio ( and vapour pressure )
    Using ITS-90 correction of Wexler's formula
    Optional enhancement factors for non ideal 
    """
    dp=np.atleast_1d(dp)
    g=np.array([-2.8365744e3,-6.028076559e3,1.954263612e1,-2.737830188e-2, 
       1.6261698e-5,7.0229056e-10,-1.8680009e-13,2.7150305],dtype='f8')
    lnes=np.log(dp)*g[7]
    for i in range(7):lnes=lnes+g[i]*(dp**(i-2.0))
    vp=np.exp(lnes)/1e2
    # if(enhance and len(p)>0) :
    #     A=np.array([-1.6302041e-1,1.8071570e-3,-6.7703064e-6,8.5813609e-9],dtype='f8')
    #     B=np.array([-5.9890467e1,3.4378043e-1,-7.7326396e-4,6.3405286e-7],dtype='f8')
    #     if(len(temp)==0) : temp=fp
    #     alpha=np.zeros(fp.shape)
    #     beta=np.zeros(fp.shape)
    #     for i in range(4) :
    #         alpha=alpha+(A[i]*(temp**i))
    #         beta=beta+(B[i]*(temp**i))
    #     beta=np.exp(beta)
    #     ef=np.exp(alpha*(1-vp/p)+beta*(p/vp-1))
    #     vp=vp*ef
    return vp

# def saturation_vapour_pressure(temp):
#     """
     
#     """
#     T0 = 273.16
#     es0 = 611.655
#     cpl_cpv = 2180
#     Rv = 461.52
#     L0 = 2.501e6
#     L = L0 - cpl_cpv * (temp - T0)

#     svp = es0 * (T0 / temp)**(cpl_cpv / Rv) * np.exp((L0 / (Rv * T0) - L / (Rv * temp)))
#     return svp

class TotalWater(PPBase):

    inputs = [
        #'CALTNOS', 'CALTSAM', 'CALTAMB', 'CALTSRC', 'CALHTR1', 'CALHTR2',
        #'CALISRC',
        'TWCDAT_twc_detector',
        'TWCDAT_twc_nose_temp',
        'TWCDAT_twc_samp_temp',
        'TWCDAT_twc_amb_temp', 
        'TWCDAT_twc_srce_temp',
        'TWCDAT_twc_evap1',
        'TWCDAT_twc_evap2',                                                  
        'TWCDAT_twc_srce_i',
        'TWCDAT_twc_evap2_on',
        'TWCDAT_status',
        'PS_RVSM',
        'TAT_DI_R',
        'WVSS2F_VMR_C'
    ]

    def declare_outputs(self):
        self.declare(
            'TWC_DET',
            units='1',
            long_name='Raw data from the TWC probe Lyman alpha detector',
            frequency=256
        )

        self.declare(
            'TNOS_CAL',
            units='K',
            long_name='Temperature of the TWC probe nose',
            frequency=1
        )

        self.declare(
            'TWC_TSAM',
            units='K',
            long_name='Temperature of the TWC probe sample',
            frequency=1
        )

        self.declare(
            'TAMB_CAL',
            units='K',
            long_name='TWC Ambient temperature',
            frequency=1
        )

        self.declare(
            'TSRC_CAL',
            units='K',
            long_name='TWC source temperature',
            frequency=1
        )

        self.declare(
            'HTR1_CAL',
            units='A',
            long_name='TWC heater 1 current',
            frequency=1
        )

        self.declare(
            'HTR2_CAL',
            units='A',
            long_name='TWC heater 2 current',
            frequency=1
        )

        self.declare(
            'ISRC_CAL',
            units='A',
            long_name='TWC source current',
            frequency=1
        )

        self.declare(
            'STAT_CAL',
            units='1',
            long_name='TWC status word',
            frequency=1
        )

    def process(self):
        # self.get_dataframe()
        # d = self.d
        # print(d)
        # caltnos = d['CALTNOS']
        # caltsam = d['CALTSAM']
        # caltamb = d['CALTAMB']
        # caltsrc = d['CALTSRC']
        # calhtr1 = d['CALHTR1']
        # calhtr2 = d['CALHTR2']
        # calisrc = d['CALISRC']

        caltnos = np.array([1.2614E-16, -1.8668E-12, 1.2704E-08, 9.4262E-03, 3.1654E+02])
        caltsam = np.array([11.02757718, 3956.07906064, 1932.77875494, 7162.35330738])
        #caltsam = np.array([2.5406E-17, 3.5253E-13, 2.1300E-09, 7.1707E-06, 2.7927E-02, 4.0343E+02])
        caltamb = np.array([2.9951E-13, 9.3288E-10, -4.0779E-06, 1.6016E-02, 2.7390E+02])
        caltsrc = np.array([2.8700E-18, -1.2794E-14,  2.8480E-11,  2.2585E-10,  9.5178E-03,  3.7298E+02])
        calhtr1 = np.array([9.7752E-04, 0.0000E+00])
        calhtr2 = np.array([9.7752E-04, 0.0000E+00])
        calisrc = np.array([9.7636E-08, -2.5957E-06])

        ranges = {
            'TWC_DET': (-9*1024, 9*1024, 20*1024),
            'TNOS_CAL': (314, 383, 10),
            'TWC_TSAM': (300, 388, 30),
            'TAMB_CAL': (289, 343, 2),
            'TSRC_CAL': (378, 393, 5),
            'HTR1_CAL': (0.3, 6.6, 0.5),
            'HTR2_CAL': (0.3, 6.6, 0.5),
            'ISRC_CAL': (-1.1e-3, -0.4e-3, 0.05e-3),
            'STAT_CAL': (0, 255, 255)
        }

        twc_det = DecadesVariable(
            self.dataset['TWCDAT_twc_detector'](),
            name='TWC_DET',
            flag=DecadesClassicFlag,
        )

        tnos_cal = DecadesVariable(
            pd_polyval(caltnos, self.dataset['TWCDAT_twc_nose_temp']()),
            name='TNOS_CAL',
            flag=DecadesClassicFlag
        )

        twc_tsam = DecadesVariable(
            # thermistor(self.dataset['TWCDAT_twc_samp_temp'](), *caltsam),
            pd_polyval(caltsam, self.dataset['TWCDAT_twc_samp_temp']()),
            name='TWC_TSAM',
            flag=DecadesClassicFlag
        )

        tamb_cal = DecadesVariable(
            pd_polyval(caltamb, self.dataset['TWCDAT_twc_amb_temp']()),
            name='TAMB_CAL',
            flag=DecadesClassicFlag
        )

        tsrc_cal = DecadesVariable(
            pd_polyval(caltsrc, self.dataset['TWCDAT_twc_srce_temp']()),
            name='TSRC_CAL',
            flag=DecadesClassicFlag
        )

        htr1_cal = DecadesVariable(
            pd_polyval(calhtr1, self.dataset['TWCDAT_twc_evap1']()),
            name='HTR1_CAL',
            flag=DecadesClassicFlag
        )

        htr2_cal = DecadesVariable(
            pd_polyval(calhtr2, self.dataset['TWCDAT_twc_evap2']()),
            name='HTR2_CAL',
            flag=DecadesClassicFlag
        )

        isrc_cal = DecadesVariable(
            pd_polyval(calisrc, self.dataset['TWCDAT_twc_srce_i']()),
            name='ISRC_CAL',
            flag=DecadesClassicFlag
        )

        stat_cal = DecadesVariable(
            self.dataset['TWCDAT_status'](),
            name='STAT_CAL',
            flag=DecadesClassicFlag
        )

        for var in (
            twc_det, tnos_cal, twc_tsam, tamb_cal,
            tsrc_cal, htr1_cal, htr2_cal, isrc_cal, stat_cal
        ):
            
            valid_min, valid_max, valid_roc = ranges[var.name]
            var.flag.add_meaning(0, 'data_good')
            var.flag.add_meaning(1, 'outside_valid_range_or_rate_of_change')

            flag = np.zeros(var.data.shape, dtype=np.uint8)
            flag[(var.data < valid_min) | (var.data > valid_max)] = 1
            roc = (var().diff().abs() / var.frequency) > valid_roc
            flag[roc] = 1
            var.flag.add_flag(flag)

            self.add_output(var) 

        # Perform the fit w/ the hygrometer data
        index = twc_tsam().index
        pressure = self.dataset['PS_RVSM']().reindex(index)
        temperature = self.dataset['TAT_DI_R']().reindex(index)
        vmr = self.dataset['WVSS2F_VMR_C']().reindex(index)
        import matplotlib.pyplot as plt
        plt.plot(twc_tsam)
        plt.title('tmp')
        plt.show()
        fit = make_fit(
            pressure, temperature,
            twc_tsam(),
            twc_det.reindex(index), vmr
        )

        ans = pd_polyval(fit, twc_det)
        plt.plot(ans)
        plt.title('answer')
        plt.show()
        p1 = self.dataset['PS_RVSM']().reindex(ans.index).interpolate()
        t2 = twc_tsam().reindex(ans.index).interpolate() 

        KO = get_KO(p1)
        # vpo = (ans - (KO * u0 * pressure / (Kv * samp_temp))) * samp_temp
        plt.plot(p1)
        plt.title('P1')
        plt.show()
        plt.plot(t2)
        plt.title('T2')
        plt.show()
        vpo = (ans - (KO * u0 * p1 / (Kv * t2))) * t2
        
        # vmro = vp2vmr(vpo, pressure)
        # mmr = vmr_mmr(vmro)
        import matplotlib.pyplot as plt
        plt.plot(vpo)
        plt.plot(vmr)
        plt.title('VMR')
        plt.show()

        twc_tdew =  pd.Series(
            vp2dp(vpo.ravel()), index=ans.index
        )
        plt.plot(twc_tdew)
        plt.plot(self.dataset['TDEWCR2C']())
        plt.show()


