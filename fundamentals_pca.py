import requests
import polars as pl
import requests
from sec_cik_mapper import StockMapper
from dotenv import dotenv_values
env = dotenv_values('.env')
agent = env['agent']
headers = {
    'User-Agent': agent
}
def getdatefundamentals(ticker, startdate, enddate):
    mapper = StockMapper()
    cik = mapper.ticker_to_cik[ticker]
    r = requests.get('https://data.sec.gov/api/xbrl/companyfacts/CIK' + cik + '.json',headers=headers)
    data = r.json()
    Metrics = {
        'EarningsPerShareBasic': ('USD/shares', 'start', 'EPS'),
        'Assets' : ('USD', 'filed', 'TotalAssets'),
        'Liabilities' : ('USD', 'filed', 'TotalLiabilities'),
        'NetIncomeLoss' : ('USD', 'start', 'NetIncome')
    }

    alldates = set()
    for gaapname, (unit, date_field, title) in Metrics.items():
        entries = data['facts']['us-gaap'][gaapname]['units'][unit]
        dates = {entry[date_field] for entry in entries}
        alldates.update(dates)

    df = pl.DataFrame({'date' : sorted(d for d in alldates if startdate < d < enddate)})
    df = df.with_columns(pl.col('date').str.to_date())
    
    for gaapname, (unit, date_field, title) in Metrics.items():
        metric_data = data['facts']['us-gaap'][gaapname]['units'][unit]
        print(metric_data)
        metric_map = {entry[date_field]: entry['val'] for entry in metric_data if date_field in entry}
        df = df.with_columns(
            pl.col('date').replace(metric_map, default=None).alias(title)
        )
    df = df.fill_null(strategy='forward')
    df = df.fill_null(strategy='backward')
    print(df)

getdatefundamentals('AAPL', '2020-01-02', '2025-12-19')
