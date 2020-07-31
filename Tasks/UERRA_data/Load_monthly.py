import cdsapi

c = cdsapi.Client()
variables=['wind_speed','wind_direction','temperature','relative_humidity','pressure']

for var in variables:
        for i in range(2019,1960,-1):
                year=str(i)
                for j in range(12,0,-1):
                        month=str(j)
                        if j<10:
                                month='0'+month
                        
                        filename='UERRA_'+var+'_'+year+'_'+month+'.grib'
                              
                        c.retrieve(
                        'reanalysis-uerra-europe-height-levels',
                        {
                                'variable': var,
                                'height_level': [
                                '15_m', '500_m',
                                ],
                                'year': year,
                                'month': month,
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
                        filename)
                        
                        
