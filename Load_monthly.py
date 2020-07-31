import cdsapi

c = cdsapi.Client()

c.retrieve(
            'reanalysis-uerra-europe-height-levels',
                {
                            'variable': 'wind_speed',
                                    'height_level': [
                                                    '15_m', '500_m',
                                                            ],
                                            'year': '2019',
                                                    'month': '02',
                                                            'day': [
                                                                            '01', '02', '03',
                                                                                        '04', '05', '06',
                                                                                                    '07', '08', '09',
                                                                                                                '10', '11', '12',
                                                                                                                            '13', '14', '15',
                                                                                                                                        '16', '17', '18',
                                                                                                                                                    '19', '20', '21',
                                                                                                                                                                '22', '23', '24',
                                                                                                                                                                            '25', '26', '27',
                                                                                                                                                                                        '28',
                                                                                                                                                                                                ],
                                                                    'time': [
                                                                                    '00:00', '06:00', '12:00',
                                                                                                '18:00',
                                                                                                        ],
                                                                            'format': 'grib',
                                                                                },
                    'download.grib')
