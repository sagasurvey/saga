from IPython.display import display_html

def show_images(table, pixscale=0.2, size=200,
                keys=('OBJID', 'RA', 'DEC', 'survey', 'r_mag', 'sb_r', 'gr', 'RHOST_KPC')):
    out = []
    for row in table:
        if row['survey'] == 'des':
            layer = 'des-dr1'
        elif row['survey'] == 'decals':
            layer = 'mzls+bass-dr6' if row['DEC'] > 32 else 'decals-dr7'
        else:
            layer = 'sdssco'

        host_id = 'nsa{}'.format(row['HOST_NSAID']) if row['HOST_NSAID'] != -1 else 'pgc{}'.format(row['HOST_PGC'])

        url = 'http://legacysurvey.org/viewer-dev/jpeg-cutout/?ra={ra}&dec={dec}&pixscale={pixscale}&layer={layer}&size={size}'.format(
            ra=row['RA'],
            dec=row['DEC'],
            layer=layer,
            pixscale=pixscale,
            size=size,
        )

        link = 'http://legacysurvey.org/viewer-dev?ra={ra}&dec={dec}&layer={layer}&zoom=16'.format(
            ra=row['RA'],
            dec=row['DEC'],
            layer=layer,
        )
        title = 'host = {}\n'.format(host_id)
        title += '\n'.join(('{} = {}'.format(k, row[k] if i < 4 else '{:.2f}'.format(row[k])) for i, k in enumerate(keys)))
        out.append('<a href="{}" target="_blank"><img src="{}" style="display:inline-block;" title="{}" /></a>'.format(link, url, title))
    display_html('\n'.join(out), raw=True)
