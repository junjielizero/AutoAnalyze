#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import scorecardpy as sc 


def filter_var_by_chitest(df):
    var = ['target']
    for col in df.columns:
        if col != 'target':
            obs = pd.crosstab(df.target, df[col]).values
            _, p_values, _, _ = chi2_contingency(obs)
            if p_values < 0.05:
                var.append(col)
    return var


def continous_bin(i, nbins=5):
    s = i[i > 0]
    _, bins = pd.qcut(s, nbins, retbins=True)
    bins[-1] = np.inf
    return bins


def cont2discret(srs, nbins=5):
    # contious to discret based on quantile
    srs = srs.copy()
    number = [*range(1,nbins+1)]
    number.sort(reverse = True)
    for i in number:
        try:
            # not null cut 
            d=pd.cut(srs[srs.notna()], continous_bin(srs, i), right=False)\
            .cat.add_categories(0).fillna(0)
            srs[d.index] = d
            break
        except ValueError:
            pass
    return srs

def qcut_bin(srs,nbins=5):
    # global out
    # global cut_bins
    # contious to discret based on qcut
    # if series min values occupy most, which srs quantile(0.6) is srs min, then cut two group which is -np.inf,srs.min(),np.inf cut.
    # if not, qcut 
    if (srs.quantile(0.6) == srs.min()) or (srs.nunique()==2):
        out,cut_bins = pd.cut(srs,[-np.inf,srs.min(),np.inf],right=True,precision=0,retbins=True)     
    else:
        number = [*range(1,nbins+1)]
        number.sort(reverse = True)
        for i in number:
            try:
                _, cut_bins =pd.qcut(srs, q=i,retbins=True,precision=0)
                cut_bins[-1] = np.inf
                out, cut_bins = pd.cut(srs, cut_bins, right=False,retbins=True)
                break
            except ValueError:
                pass
    return out,cut_bins


def df_col2woebins(df,bins,target='target'):
    df = df.copy()

    ncols = df.select_dtypes(include='number').columns.drop(target)
    scols = df.select_dtypes(exclude='number').columns
    for k in ncols:
        cut =  [float(i) if i!='inf' else np.inf for i in bins[k]['breaks'].values ]
        cut.insert(0,-np.inf)
        df[k] = pd.cut(df[k],cut,right=False,labels=bins[k]['woe']).astype(float)
    for k in scols:
        dct = {a:b for a,b in bins[k][['bin','woe']].values}
        df[k] = df[k].map(dct)

    return df
  

def iv_cls(bins, cols,target='target'):
    d = {i: bins[i]['total_iv'][0] for i in cols if i != target}
    df = pd.DataFrame\
        .from_dict(d, orient='index').sort_values(by=0, ascending=0)\
        .rename(columns={0: 'value'})
    idx = df.query('value>=0.02').index
    return df, idx


def cal_woe_iv(dataset,feature,target='target'):
    # calculate woe,bin_iv,iv etc.,return scorecardpy bins
    # ref:https://towardsdatascience.com/attribute-relevance-analysis-in-python-iv-and-woe-b5651443fc04
    dataset = dataset.copy()
    if pd.api.types.is_object_dtype(dataset[feature])==False:
        dataset[feature], cut_bins = qcut_bin(dataset[feature]) 
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'bin': val,
            'count': dataset[dataset[feature] == val].count()[feature],
            'good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        })
        
    dset = pd.DataFrame(lst)
    dset['variable'] = feature
    dset['count_distr'] = dset['count']/dset['count'].sum()
    dset['Distr_Good'] = dset['good'] / dset['good'].sum()
    dset['Distr_Bad'] = dset['bad'] / dset['bad'].sum()
    dset['badprob'] = dset['bad']/dset['count']
    dset['woe'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'woe': {np.inf: 0, -np.inf: 0}})
    dset['bin_iv'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['woe']
    dset['is_special_values'] = 'False'
    dset['total_iv'] = dset['bin_iv'].sum()   
    dset = dset.sort_values(by='bin')
    dset['bin'] = dset['bin'].astype(str)
    if pd.api.types.is_object_dtype(dataset[feature])==False:
        dset['breaks'] = cut_bins[1:]
    else:
        dset['breaks'] = dset['bin']
    dset = dset[['variable', 'bin', 'count', 'count_distr', 'good', 'bad', 'badprob',
       'woe', 'bin_iv', 'total_iv','breaks','is_special_values']]
    dset.index = range(len(dset))
    return dset

def woe_bin_fix(df,y='target'):
    df = df.copy()
    bins = sc.woebin(df,y=y
                   ,min_perc_coarse_bin=0.1
                   ,method='chimerge')
    iv_df, _=iv_cls(bins,df.columns.tolist()[1:])
    # fix zero iv var
    cols = iv_df.query('value==0').index 
    for c in cols:
        bins[c] = cal_woe_iv(df,c,target=y)
    iv_df, _=iv_cls(bins,df.columns.tolist()[1:])
    return bins, iv_df

if __name__ == '__main__':
    continous_bin()
    cont2discret()
    filter_var_by_chitest()
    qcut_bin()
    df_col2woebins()
    woe_bin_fix()
