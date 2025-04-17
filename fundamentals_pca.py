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

Metrics = {
        'EarningsPerShareBasic': ('USD/shares', 'start', 'EPS'),
        'Assets' : ('USD', 'filed', 'TotalAssets'),
        'Liabilities' : ('USD', 'filed', 'TotalLiabilities'),
        'NetIncomeLoss' : ('USD', 'start', 'NetIncome')
    }

def getdatefundamentals(ticker, startdate, enddate):
    try:   
        mapper = StockMapper()
        cik = mapper.ticker_to_cik[ticker]
        r = requests.get('https://data.sec.gov/api/xbrl/companyfacts/CIK' + cik + '.json',headers=headers)
        data = r.json()

        alldates = set()
        for gaapname, (unit, date_field, title) in Metrics.items():
            entries = data['facts']['us-gaap'][gaapname]['units'][unit]
            dates = {entry[date_field] for entry in entries}
            alldates.update(dates)

        df = pl.DataFrame({'date' : sorted(d for d in alldates if startdate < d < enddate)})
        df = df.with_columns(pl.col('date').str.to_date())
        
        for gaapname, (unit, date_field, title) in Metrics.items():
            try:
                metric_data = data['facts']['us-gaap'][gaapname]['units'][unit]
                metric_map = {entry[date_field]: entry['val'] for entry in metric_data if date_field in entry}
                df = df.with_columns(
                    pl.col('date').replace_strict(metric_map, default=None).alias(title)
                )
            except KeyError:
                df = df.with_columns(pl.lit(None).alias(title))
        return df.with_columns(pl.lit(ticker).alias('ticker'))
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return None

startdate = '2015-04-07'
enddate = '2025-04-04'
holdings = pl.read_csv('holdings.csv')
dfs = []
for ticker in holdings['Holding Ticker']:
    print(ticker)
    df = getdatefundamentals(ticker.strip(), startdate, enddate)
    dfs.append(df)

dfs = [df for df in dfs if df is not None]
value_columns = [metric[2] for metric in Metrics.values()]

combined = pl.concat(dfs).pivot(
    values= value_columns,
    index= 'date',
    columns= 'ticker',
    aggregate_function= 'first'
)
combined = combined.fill_null(strategy='backward')
combined = combined.fill_null(strategy='forward')
print(combined)
combined.write_csv('fundamentals_data.csv')