import requests
import polars as pl
import json
import requests
def getfundamentals(ticker, timeframe):
    CIK = getCIK(ticker)
    r = requests.get('https://data.sec.gov/api/xbrl/companyfacts/CIK'+ CIK + '.json')
    json_data = r.json()
    df = pl.from_dicts(json_data)
    filtered_df = df.select([
        pl.col('facts.us-gaap')
    ])
    #load ticker json fom CIK
    #what fundamentals do you want
    #for every date in timeframe look for closest set of fundamentals
    #