# Source: https://gist.github.com/pierdom/235278048ab7127b64c1f87ca7297c87
# import matplotlib and set inline
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as st
from data.data_handler import get_data
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import os

# TODO: Check Kolmogorov-Smirnov Goodness-of-Fit Test:
#  https://www.itl.nist.gov/div898/handbook/eda/section3/eda35g.htm
# TODO: Another notebook:
#  https://github.com/amirarsalan90/dist_fitting_medium/blob/master/dist_fitting.ipynb, corresponding
#  medium-article: https://medium.com/@amirarsalan.rajabi/distribution-fitting-with-python-scipy-bb70a42c0aed

# Set seaborn style
sns.set_style("whitegrid")

# The distribution
df = get_data('01.01.2014', '31.12.2018', ['System Price'], os.getcwd())
df['Date'] = df['Date'] + pd.to_timedelta(df['Hour'], unit='h')
df.drop(['Hour'], axis=1, inplace=True)
df.set_index(keys='Date', drop=True, inplace=True)
df.dropna(subset=['System Price'], inplace=True)  # Maybe interpolate? Change to: .interpolate()
mydistr = df['System Price'].values

# X axis with custom binning (change here, depending on what you need)
mydistr_x = np.linspace(mydistr.min(), mydistr.max(), len(mydistr))
_ = sns.distplot(mydistr, bins=mydistr_x, kde=False)

# define a set of distributions to check, st.frechet_r, st.frechet_l were depreciated since 1.0.
distribution_list = [
    st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2, st.cosine,
    st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f, st.fatiguelife, st.fisk,
    st.foldcauchy, st.foldnorm, st.genlogistic, st.genpareto, st.gennorm, st.genexpon,
    st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz, st.gumbel_r,
    st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant, st.invgamma, st.invgauss,
    st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign, st.laplace, st.levy, st.levy_l, st.levy_stable,
    st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke, st.nakagami, st.ncx2, st.ncf,
    st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm, st.rdist, st.reciprocal,
    st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t, st.triang, st.truncexpon, st.truncnorm,
    st.tukeylambda,
    st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy
]

# this list will contain the result of the fittings
distribution_fitting = []

# fit every distribution in the list above
for ref_distr in distribution_list:

    # pick distribution name
    distr_name = type(ref_distr).__name__.split("_")[0]

    try:
        # ignore warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # get parameters from the best fit
            params = ref_distr.fit(mydistr)
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            # build the PDF from previous parameters
            pdf_fitted = ref_distr.pdf(mydistr_x, loc=loc, scale=scale, *arg)

            # calculate maximum likelihood estimator
            mle = ref_distr.nnlf(params, mydistr)

            # as an alternative: sum of square error
            sse = np.sum(np.power(mydistr_x - pdf_fitted, 2.0))

            # add results to list
            distribution_fitting.append({
                "distr_name": distr_name, "mle": mle, "sse": sse,
                "pdf_fitted": pdf_fitted, "params": params})

        # ignore distributions that could not be fitted
    except Exception:
        print("Discarded function: {}".format(distr_name))

# plot data histograms
fig, ax = plt.subplots(1, 1, dpi=120)
_ = ax.hist(mydistr, mydistr_x, density=True, color="gray", alpha=0.4)

# set stuff for this axis
top = 5
ax.set_prop_cycle('color', plt.cm.rainbow(np.linspace(0, 1, top)))  # plt.cm.rainbow(np.linspace(0, 1, top))
ax.set_title("Best fitting PDFs")

# just print TOP distributions according to maxim. likelihood estimator

sort_by = "mle"  # or sse
for d in sorted(distribution_fitting, key=lambda k: k[sort_by])[:top]:
    distr_name = d['distr_name']
    mle = d['mle']
    pdf_fitted = d['pdf_fitted']
    params = d['params']

    label = "{} [{:.2f}]".format(distr_name, mle)
    _ = ax.plot(mydistr_x, pdf_fitted, label=label)

_ = ax.legend()

plt.savefig('distributions.png')
