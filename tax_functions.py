import re
import pandas as pd
import numpy as np
from pandas import IndexSlice as idx
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import njit
import os

patterns_dict = {
    'instrument & transaction': {'pattern': r'(.+) (CONS|SDRT|COMM|DIVIDEND)', 'group names': ('instrument', 'transaction type')},
    'quantity & price': {'pattern': r'(\d+)@(\d+\.\d+|\d+)', 'group names': ('quantity', 'price')},
    'trade id': {'pattern': r':(\d+~\d+)', 'group names': ('trade id',)}
    }


instrument_rename_dict = {
    'SEPLAT Energy PLC':'SEPLAT Petroleum Development Co Plc',
    'Royal Bank of Scotland Group PLC':'NatWest Group PLC'
    }


header_f = lambda x: '<b><u>' + x + '</u></b>'
subheader_f = lambda x: '<b>' + x + '</b>'
curr_f = lambda x: '£{:,.2f}'.format(x)
abs_curr_f = lambda x: curr_f(np.abs(x))
fx_f = lambda x: '{:,.2f}'.format(x)
q_f = lambda x: '{:,.0f}'.format(x)
abs_q_f = lambda x: q_f(np.abs(x))



@njit
def calc_avg_purchase_price(quantity:np.array, value:np.array) -> np.array:
    nrows, = quantity.shape
    avg_prices = np.empty(nrows, dtype=np.float64)
    l_cum_q = 0
    l_avg_price = 0
    #value = np.abs(value)
    for i in range(nrows):
        q = quantity[i]
        if i == 0:
            avg_price =  value[i]/q
        else:
            if q>0:
                avg_price = (l_cum_q*l_avg_price + value[i])/(l_cum_q+q)

        avg_prices[i] = avg_price
        l_avg_price = avg_price
        l_cum_q += q
    return avg_prices


def to_lower_with_spaces(text:str) -> str:
    p = re.compile(r'([A-Z]{1}[a-z]+)')
    mo:list = p.findall(text)
    return (' '.join(mo)).lower()


def read_csv(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return setup_df(df)


def setup_df(df:pd.DataFrame) -> pd.DataFrame:
    df.columns = [to_lower_with_spaces(x) for x in df.columns]
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df = df.infer_objects()
    return df


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_title(s):
    return re.sub(r"(?:(?<=\W)|^)\w(?=\w)", lambda x: x.group(0).upper(), s)


def reformat_add_total(df:pd.DataFrame, cols_format_dict:dict, subheader:str) -> str:
    """it shouldn't have to know too much about the input columns"""
    output_text = ''
    if len(df) > 0:
        cols = list(cols_format_dict.keys())
        all_numeric_cols = df.columns[df.dtypes.apply(pd.api.types.is_numeric_dtype)]
        total = df[all_numeric_cols].sum()
        price_cols = [col for col in cols if 'price' in col]
        for col in price_cols:
            total[col] = (df[col]*df['quantity']).sum()/total['quantity']
        total = total.to_frame('Total/Average').transpose()
        df = df[cols]
        total = total[cols]
        df.index = df.index.strftime('%d %b-%y')
        df = pd.concat([df, total])
        last_row = pd.IndexSlice[df.index[df.index == "Total/Average"], :]
        df.columns = [make_title(x) for x in df.columns]
        df_style = df.style.format(formatter={make_title(x):y for x,y in cols_format_dict.items()})
        df_style = df_style.applymap(lambda x: 'font-weight: bold', subset=last_row)
        df_style = df_style.set_table_styles([
                {'selector' : '','props' : [('border','1px solid')]}, 
                {"selector": "tbody td", "props": [("border", "1px solid")]}, 
                {"selector": "th", "props": [("border", "1px solid")]}
                ])

        output_text += '<p>'+subheader_f(subheader)
        output_text += df_style.to_html()
    return output_text


def get_FX() -> pd.DataFrame:
    fx = pd.read_excel(r"Inputs\FX.xlsx", sheet_name='Rates', index_col=0)
    fx = fx.dropna(how='all')
    return fx


def write_summary(summary:dict) -> str:
    text = '<table>'
    for k,v in summary.items():
        text += '<tr>'
        text += '<td>'+k+'</td>'
        text += '<td>'+v+'</td>'
        text += '</tr>'

    text += '</table>'
    text += '<br><br>'
    return text


def process_trades(trades:pd.DataFrame) -> pd.DataFrame:
    trades['cum. quantity'] = trades.groupby(level=0)['quantity'].cumsum()
    trades['position id'] = (1*(trades['cum. quantity']==0)).groupby(level=0).shift(1, fill_value=0)
    trades['position id'] = trades['position id'].groupby(level=0).cumsum()
    trades['WMA purchase gross price'] = trades.groupby(level=0).apply(calc_WMA_per_share, col_name = 'gross value')
    trades['WMA purchase net price'] = trades.groupby(level=0).apply(calc_WMA_per_share, col_name = 'net value')
    trades['WMA allowable costs per share'] = trades.groupby(level=0).apply(calc_WMA_per_share, col_name = 'allowable costs')
    sell_quantity = np.abs(trades['quantity'].where(trades['sell?']))
    trades['total allowable costs'] = -trades['WMA purchase net price'] * sell_quantity + trades['allowable costs']
    trades['gross P&L'] = (trades['price']- trades['WMA purchase gross price']) * sell_quantity
    trades['net P&L'] = (trades['price']- trades['WMA purchase net price']) * sell_quantity + trades['allowable costs']
    trades['realised allowable costs'] = trades['allowable costs'] + sell_quantity*trades['WMA allowable costs per share'].fillna(0)
    trades['implied realised allowable costs'] = np.abs(trades['gross P&L'] - trades['net P&L'])
    return trades


def process_trades_old(trades:pd.DataFrame) -> pd.DataFrame:
    trades['cum. quantity'] = trades.groupby(level=0)['quantity'].cumsum()
    trades['position id'] = (1*(trades['cum. quantity']==0)).groupby(level=0).shift(1, fill_value=0)
    trades['position id'] = trades['position id'].groupby(level=0).cumsum()
    trades['WMA purchase gross price'] = trades.groupby(level=0).apply(calc_WMA_per_share, col_name = 'gross value')
    trades['WMA purchase net price'] = trades.groupby(level=0).apply(calc_WMA_per_share, col_name = 'net value')
    trades['WMA allowable costs per share'] = trades.groupby(level=0).apply(calc_WMA_per_share, col_name = 'allowable costs')
    sell_quantity = np.abs(trades['quantity'].where(trades['sell?']))
    trades['gross P&L'] = (trades['price']- trades['WMA purchase gross price']) * sell_quantity
    trades['net P&L'] = (trades['price']- trades['WMA purchase net price']) * sell_quantity + trades['allowable costs']
    trades['realised allowable costs'] = trades['allowable costs'] + sell_quantity*trades['WMA allowable costs per share'].fillna(0)
    trades['implied realised allowable costs'] = np.abs(trades['gross P&L'] - trades['net P&L'])
    return trades


def create_capital_gains_report(trades:pd.DataFrame, start:pd.Timestamp, end:pd.Timestamp, name:str) -> str:
    period_trades = trades[trades['sell?']].loc[idx[:, start:end], :]# ['position id', 'quantity', 'price', 'WMA purchase gross price', 'gross P&L', 'allowable costs', 'realised allowable costs', 'net P&L']]
    summary = {'Name': name,
        'Total Realised Sales': '£{:,.2f}'.format(np.abs(period_trades[['price', 'quantity']].product(axis=1).sum())),
        'Total Allowable Costs': '£{:,.2f}'.format(period_trades['total allowable costs'].sum()),
        'Total Taxable Gains': '£{:,.2f}'.format(period_trades['net P&L'].where(period_trades['net P&L']>0).sum()),
        'Total Taxable Losses': '£{:,.2f}'.format(period_trades['net P&L'].where(period_trades['net P&L']<=0).sum()),
        'Total Net Taxable Gains/Losses': '£{:,.2f}'.format(period_trades['net P&L'].sum()), 
        }
    text = '<html><body>'
    text += '<p>' + header_f('Realised Capital Gains between {} to {}'.format(start.strftime('%d %b-%y'), end.strftime('%d %b-%y')))
    text += write_summary(summary)
    
    trades['allotted purchase price'] = trades['WMA purchase net price']
    buy_display_cols = {'price': curr_f, 'quantity': abs_q_f, 'gross value': abs_curr_f, 'allowable costs': abs_curr_f, 'net value': abs_curr_f}
    sell_display_cols = {'price': curr_f, 'allotted purchase price': curr_f, 'quantity': abs_q_f, 'gross value': abs_curr_f, 'total allowable costs': curr_f, 'net P&L': curr_f}

    for instr, pos_id in period_trades.groupby(['instrument', 'position id']).max().index:
        instr_pos_trades = trades.xs(instr)
        instr_pos_trades = instr_pos_trades[(instr_pos_trades['position id'] ==  pos_id)]
        buys = instr_pos_trades[instr_pos_trades['buy?']].loc[:end, :]
        sells = instr_pos_trades[instr_pos_trades['sell?']].loc[start:end, :]
        sells_before_this_period = instr_pos_trades[instr_pos_trades['sell?']].loc[:(start-pd.Timedelta('1d')), :]
        text += '<p>'+header_f('{} ({})'.format(instr, pos_id))
        text += reformat_add_total(sells, sell_display_cols, 'Sells')
        text += reformat_add_total(buys, buy_display_cols, 'Buys')
        text += reformat_add_total(sells_before_this_period, sell_display_cols, 'Sales in Prior Tax Years')
        text += '<br><br>'
    text += '</body></html>'
    return text


def create_capital_gains_report_for_me(trades:pd.DataFrame, start:pd.Timestamp, end:pd.Timestamp, name:str) -> str:
    period_trades = trades[trades['sell?']].loc[idx[:, start:end], ['position id', 'quantity', 'price', 'WMA purchase gross price', 'gross P&L', 'allowable costs', 'realised allowable costs', 'net P&L']]
    summary = {'Name': name,
        'Total Realised Sales': '£{:,.2f}'.format(np.abs(period_trades[['price', 'quantity']].product(axis=1).sum())),
        'Realised Allowable Costs': '£{:,.2f}'.format(np.abs(period_trades['realised allowable costs'].sum())),
        'Total Taxable Gains': '£{:,.2f}'.format(period_trades['net P&L'].where(period_trades['net P&L']>0).sum()),
        'Total Taxable Losses': '£{:,.2f}'.format(period_trades['net P&L'].where(period_trades['net P&L']<=0).sum()),
        'Total Net Taxable Gains/Losses': '£{:,.2f}'.format(period_trades['net P&L'].sum()), 
        }
    text = '<html><body>'
    text += '<p>' + header_f('Realised Capital Gains between {} to {}'.format(start.strftime('%d %b-%y'), end.strftime('%d %b-%y')))
    text += write_summary(summary)
    
    buy_display_cols = {'price': curr_f, 'quantity': abs_q_f, 'gross value': abs_curr_f, 'allowable costs': abs_curr_f, 'net value': abs_curr_f}
    sell_display_cols = {'price': curr_f, 'WMA purchase net price': curr_f, 'quantity': abs_q_f, 'gross value': abs_curr_f, 'allowable costs': curr_f, 'net value': abs_curr_f, 'realised allowable costs': curr_f, 'net P&L': curr_f}

    for instr, pos_id in period_trades.groupby(['instrument', 'position id']).max().index:
        instr_pos_trades = trades.xs(instr)
        instr_pos_trades = instr_pos_trades[(instr_pos_trades['position id'] ==  pos_id)]
        buys = instr_pos_trades[instr_pos_trades['buy?']].loc[:end, :]
        sells = instr_pos_trades[instr_pos_trades['sell?']].loc[start:end, :]
        sells_before_this_period = instr_pos_trades[instr_pos_trades['sell?']].loc[:(start-pd.Timedelta('1d')), :]
        text += '<p>'+header_f('{} ({})'.format(instr, pos_id))
        text += reformat_add_total(sells, sell_display_cols, 'Sells')
        text += reformat_add_total(buys, buy_display_cols, 'Buys')
        text += reformat_add_total(sells_before_this_period, sell_display_cols, 'Sales in Prior Tax Years')
        text += '<br><br>'
    text += '</body></html>'
    return text



def calc_WMA_per_share(df:pd.DataFrame, col_name:str) -> pd.Series:
    return pd.Series(calc_avg_purchase_price(df['quantity'].values, 
                        df[col_name].values), 
                        index=df.index.get_level_values(1))

