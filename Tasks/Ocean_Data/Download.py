import cdsapi

c = cdsapi.Client()

c.retrieve(
    'satellite-sea-level-mediterranean',
    {
        'variable': 'all',
        'format': 'zip',
        'year': '2019',
        'month': '02',
        'day': '15',
    },
    'download.zip')